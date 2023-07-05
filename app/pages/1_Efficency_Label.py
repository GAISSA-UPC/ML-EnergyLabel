import streamlit as st
import pandas as pd
import ast
from scipy import stats
import fitz
import yaml
import re
from label_generation import EnergyLabel
from energy_label import read_df_processed, load_boundaries, load_ref_metrics
import numpy as np
from data import read_data, read_sheet
from datetime import datetime
from google.oauth2.service_account import Credentials
import gspread
import streamlit_nested_layout

if 'clicks' not in st.session_state:
    st.session_state['clicks'] = {'New Model': False, 'Dataset': False}

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']


creds = Credentials.from_service_account_info(st.secrets['gcp_service_account'], scopes=scope)    
client = gspread.authorize(creds)
sheet = client.open("HFStreamlit").sheet1



with open('metadata/tags_metadata.yaml') as file:
    tags_metadata = yaml.safe_load(file)

def click_unclick(key):
    """Helper function to unclick the other button when one is clicked"""
    if key == 'New Model':
        st.session_state.clicks['New Model'] = True
        st.session_state.clicks['Dataset'] = False
    else:
        st.session_state.clicks['Dataset'] = True
        st.session_state.clicks['New Model'] = False

def unclick(key):
    """Helper function to unclick the other button when one is clicked"""
    st.session_state.clicks[key] = False

def standardize_name(name: str) -> str:
    """Helper function to standardize the name of the model"""
    return name.lower().replace("_", " ")


st.set_page_config(
    page_title="Efficency Label Generation",
    page_icon=":bar_chart:",
)

st.markdown("<h1 style='text-align: center;'>Efficiency Label Generator</h1>", unsafe_allow_html=True)

@st.cache_data
def performance_to_dict(df):
    """Helper function to convert the performance metrics from string to dictionary"""
    df['performance_metrics'] = df['performance_metrics'].apply(lambda metrics_dict:  ast.literal_eval(metrics_dict.replace('nan', 'None')) if isinstance(metrics_dict, str) else metrics_dict)
    return df

df = read_data(sheet)
df = performance_to_dict(df)
df_co2 = df[df['co2_reported']==True]


metrics_ref = load_ref_metrics(df_co2)
boundaries = load_boundaries(df_co2, list(metrics_ref.keys()), metrics_ref)

df_co2 = df_co2.drop_duplicates(subset='modelId', keep='first')
df_co2 = df_co2[df_co2['co2_eq_emissions'] > 0]

st.markdown(
    """ 
    Explore our interactive tool for estimating the energy efficiency of your machine learning model.
    Submit details about your model to generate an energy label, with 'A' being the most energy-efficient
    and 'E' the least. 
    
    These labels are relative to all other models in the Hugging Face Hub,
    and are calculated considering factors such as CO2 emissions, model size, reusability, and performance. You can also compare your model's efficiency label with others in our database.
     
    You can estimate the CO2 emissions using tools such as [code carbon](https://codecarbon.io/) or [MLCO2 calculator](https://mlco2.github.io/impact/).
    """
)




def min_max_normalize(series):
    """Normalizes a series between 0 and 1"""
    return (series - series.min()) / (series.max() - series.min())


def string_to_dict(s):
    """
    Converts a string of the form 'key1:value1,key2:value2,...' to a dictionary {key1: value1, key2: value2, ...}
    
    Args:
        s (str): string to be converted to dictionary
    
    Returns:
        dict: dictionary with the keys and values from the string
    """

    if s == '':
        return {}
    d = {}
    pairs = s.split(',')
    for pair in pairs:
        key, value = pair.split(':')
        d[key] = float(value)  
    return d


def performance_score(metrics_dict, df):
    """
    Calculates the performance score of a model based on the performance metrics of the model and the performance metrics of the other models in the database.

    Args:
        metrics_dict (dict): dictionary with the performance metrics of the model
        df (pd.DataFrame): dataframe with the performance metrics of all models in the database
    
    Returns:
        float: performance score of the model
    """

    if metrics_dict == '' or metrics_dict is None:
        return None
    
    metrics_summary = {metric: {'min': df.loc[df['performance_metrics'].apply(lambda x: metric in x), 'performance_metrics'].apply(lambda x: x[metric]).min(),
                                'max': df.loc[df['performance_metrics'].apply(lambda x: metric in x), 'performance_metrics'].apply(lambda x: x[metric]).max()} 
                    for metric in metrics_dict.keys()}
    
    normalized_metrics = [(value - metrics_summary[metric]['min']) / (metrics_summary[metric]['max'] - metrics_summary[metric]['min'])
                           for metric,value in metrics_dict.items() if not pd.isnull(value)]
    
    return stats.hmean([metric for metric in normalized_metrics if not pd.isnull(metric)])


def check_other_metrics(other_metrics):
    """
    Checks if the format of the other metrics is correct. The format should be 'metric1:value1, metric2:value2,...'
    
    Args:
        other_metrics (str): string with the other metrics

    Returns:
        bool: True if the format is correct, False otherwise
    """
    
    pattern = r'^(\w+:[\w\.]+)(,\w+:[\w\.]+)*$'
    return bool(re.fullmatch(pattern, other_metrics.replace(' ', '')))

def unit_transformation(value, unit):
    """
    Transforms the value to the unit introduced.

    Args:
        value (float): value to be transformed
        unit (str): unit to which the value will be transformed

    Returns:
        float: value transformed to the unit introduced
    """

    match unit:
        case 'kg' | 'kB': 
            value = round(float(value)*1000,2)
        case 't' | 'MB':
            value = round(float(value)*1000000,2)
        case 'GB':
            value = round(float(value)*1000000000,2)

    return value

def check_attribute(attribute, value, container):
    """
    Checks if the value introduced for the attribute is correct.
    
    Args:
        attribute (str): attribute to be checked
        value (str): value introduced for the attribute
        container (st.container): container where the warning will be displayed

    Returns:
        bool: True if the value is correct, False otherwise
    """

    if value == '':
        return True
    if attribute == 'other metrics':
        if not check_other_metrics(value):
            container.warning('The format of the other metrics is not correct. Use the following format: metric1\:value1, metric2\:value2,...')
            return False
        else:
            return True
    if not value.replace('.', '', 1).replace(',', '', 1).lstrip('-').isdigit():
        container.warning(f'The {attribute.replace("_", " ")} must be a number.')
        return False
    if float(value) < 0:
        container.warning(f'The {attribute} must be non-negative.')
        return False
    elif attribute in ['accuracy', 'f1'] and not(0 <= float(value) <= 1):
        container.warning(f'The {attribute} must be between 0 and 1.')
        return False

    return True

def convert_commas_to_dots(value):
    """Converts commas to dots in a string."""
    if value == '':
        return value

    value = value.replace(",", ".")
    return value.replace(".", "", value.count(".") -1)

def gather_numeric_unit_attributes_input(attribute, units):
    """
    Gathers the value and unit of a numeric input attribute.

    Args:
        attribute (str): attribute to be gathered
        units (list): list of units to be displayed in the selectbox

    Returns:
        tuple: tuple with the value, warning (where a warning would be placed in case of incorrect value)
               and unit of the attribute
    """
    
    
    col1,col2 = st.columns([6,1])

    attribute_value = col1.text_input(attribute)
    attribute_warning = col1.empty()
    attribute_unit = col2.selectbox("Unit", units, key=f'{attribute}_unit')

    return convert_commas_to_dots(attribute_value), attribute_warning, attribute_unit
     
def gather_performance_attributes_input():
    """
    Gathers input attributes related to the performance of the model.

    Returns:
        tuple: tuple with the accuracy, warning (where a warning would be placed in case of incorrect accuracy),
                f1, warning (where a warning would be placed in case of incorrect f1) and other metrics
    """                

    col1,col2,col3 = st.columns([1,1,3])
    accuracy, f1, other_metrics = (convert_commas_to_dots(col1.text_input("Accuracy")), 
                                    convert_commas_to_dots(col2.text_input("F1")), 
                                    col3.text_input("Other Metrics (comma separated)"))
    accuracy_warning, f1_warning, other_metrics_warning = col1.empty(), col2.empty(), col3.empty()
    return accuracy, accuracy_warning, f1, f1_warning, other_metrics, other_metrics_warning

def create_main_form_fields():
    """
    Creates the main form fields.
    
    Returns:
        values_dict: dictionary with the values of the main form fields
        warnings_dict: dictionary with the warnings of the main form fields
        units_dict: dictionary with the units of the main form fields
    """

    co2, co2_warning, co2_unit = gather_numeric_unit_attributes_input('CO2 emissions', ['g', 'kg', 't'])
    model_file_size, model_file_size_warning, model_file_size_unit = gather_numeric_unit_attributes_input('Model File Size', ['B', 'kB', 'MB', 'GB'])
    datasets_size, datasets_size_warning, dataset_size_unit = gather_numeric_unit_attributes_input('Dataset Size', ['B', 'kB', 'MB', 'GB'])
    accuracy, accuracy_warning, f1, f1_warning, other_metrics, other_metrics_warning = gather_performance_attributes_input()

    downloads, downloads_warning = convert_commas_to_dots(st.text_input("Downloads")), st.empty()

    values_dict = {'co2_eq_emissions': co2, 'size': model_file_size, 'datasets_size': datasets_size, 'accuracy': accuracy, 'f1': f1, 'other_metrics': other_metrics, 'downloads': downloads}
    warnings_dict = {'co2_eq_emissions': co2_warning, 'size': model_file_size_warning, 'datasets_size': datasets_size_warning, 'accuracy': accuracy_warning, 'f1': f1_warning, 'other_metrics': other_metrics_warning, 'downloads': downloads_warning}
    units_dict = {'co2_eq_emissions': co2_unit, 'size': model_file_size_unit, 'datasets_size': dataset_size_unit}

    return values_dict, warnings_dict, units_dict


def gather_environment_input():
    """
    Gathers input attributes related to the environment of the model.
    
    Returns:
        environment_dict: dictionary with the values of the environment attributes
        environment_selectboxes: dictionary with the selectboxes of the environment attributes
        environment_others: dictionary with the attributes that allow 'Other' as an option
    """

    with st.expander("Environment"):
        st.markdown("<p style='text-align: center;'>CPUs</p>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top: 0; margin-bottom: 0'>", unsafe_allow_html=True)
        num_cores = st.text_input("Number of cores")
        num_cores_warning = st.empty()
        cpu_selectbox = st.empty()
        cpu_other = st.empty()
        st.markdown("<p style='text-align: center;'>GPUs</p>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top: 0; margin-bottom: 0'>", unsafe_allow_html=True)
        num_gpus = st.text_input("Number of GPUs")
        num_gpus_warning = st.empty()
        gpu_selectbox = st.empty()
        gpu_other = st.empty()
        st.markdown("<hr style='margin-top: 0; margin-bottom: 0'>", unsafe_allow_html=True)
        ram = st.text_input("Memory Available (GB)")
        ram_warning = st.empty()

    
    environment = {'num_cores': num_cores, 'num_gpus': num_gpus, 'ram': ram}
    environment_warning = {'num_cores': num_cores_warning, 'num_gpus': num_gpus_warning, 'ram': ram_warning}
    environment_selectboxes = {'cpu_selectbox': cpu_selectbox, 'gpu_selectbox': gpu_selectbox, 'gpu_other': gpu_other}
    environment_other_options = {'cpu_other': cpu_other, 'gpu_other': gpu_other}
    return environment, environment_warning, environment_selectboxes, environment_other_options

def gather_location_input():
    """
    Gathers input attributes related to the location of the model.
    
    Returns:
        continent_selectbox: future selectbox container for the continent
        region_selectbox: future selectbox container for the region
        region_other: future container for 'Other' option for the region
    """

    with st.expander("Geographical Location"):
        continent_selectbox = st.empty()
        region_selectbox = st.empty()
        region_other = st.empty()
    return continent_selectbox, region_selectbox, region_other


def create_additional_form_fields():
    """
    Creates the additional form fields.
    
    Returns:
        additional_values: dictionary with the values of the additional form fields
        additional_selectboxes: dictionary with the selectboxes of the additional form fields
        additional_others: dictionary with the attributes that allow 'Other' as an option
    """

    with st.expander("Additional Information (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input("Model name")
            datasets = st.text_input("Datasets (sep. by commas)")
            domain_selectbox, domain_other= st.empty(), st.empty()
            library_multiselect, library_other = st.empty(), st.empty()

        with col2:
            source_selectbox, source_other= st.empty(), st.empty()
            st.text(""); st.text("")
            continent_selectbox, region_selectbox, region_other = gather_location_input()
            st.text("")
            environment, environment_warnings, environment_selectboxes, environmental_other_options = gather_environment_input()
            training_selectbox, training_other = st.empty(), st.empty()

    additional_values = {'modelId': model_name, 'datasets': datasets, 'environment': environment}
    additional_selectboxes = {'domain_selectbox': domain_selectbox, 'library_multiselect': library_multiselect, 'source_selectbox': source_selectbox, 'training_selectbox': training_selectbox, 'region_selectbox': region_selectbox, 'continent_selectbox': continent_selectbox}
    additional_other_options = {'domain_other': domain_other, 'library_other': library_other, 'source_other': source_other, 'training_other': training_other, 'region_other': region_other}
    
    additional_selectboxes |= environment_selectboxes
    additional_other_options |= environmental_other_options

    return additional_values, environment_warnings, additional_selectboxes, additional_other_options

def create_selectbox_with_other(selectbox, other_option, df, column_name):
    """
    Creates a selectbox with an 'Other' option.
    
    Args:
        selectbox: streamlit container for the selectbox
        other_option: streamlit container for the 'Other' option
        df: dataframe with the data
        column_name: name of the column to be used as options for the selectbox

    Returns:
        attribute_selected: the final attribute selected in the selectbox
    """

    with selectbox:
        attribute_selected = st.selectbox(column_name.capitalize(), [''] + list(df[column_name].unique()) + ['Other'])
    with other_option:
        if attribute_selected == 'Other':
            attribute_selected = st.text_input(f'Select the other {column_name.capitalize()}...')
    return attribute_selected

def create_multiselect_with_other(multiselect, other_option, default_options, plural_column_name):
    """
    Creates a multiselect with an 'Other' option.
    
    Args:
        multiselect: streamlit container for the multiselect
        other_option: streamlit container for the 'Other' option
        default_options: list of default options for the multiselect
        plural_column_name: name of the column to be used as options for the multiselect

    Returns:
        attributes_selected: the final attributes selected in the multiselect
    """

    with multiselect:
        attributes_selected = st.multiselect(plural_column_name.capitalize(), default_options)
    with other_option:
        if 'Other' in attributes_selected:
            another_attributes = st.text_input(f"Enter your {plural_column_name.lower()} separated by commas...")
            if another_attributes:
                default_options = default_options + another_attributes.split(',')
                attributes_selected = attributes_selected + another_attributes.split(',')
                attributes_selected.remove("Other")
                attributes_selected = multiselect.multiselect(label=plural_column_name, default=attributes_selected, options=default_options)


def gather_attributes_with_other(additional_selectboxes, additional_other_options):
    """
    Gathers the attributes that allow 'Other' as an option.
    
    Args:
        additional_selectboxes: dictionary with the selectboxes of the additional form fields
        additional_others: dictionary with the attributes that allow 'Other' as an option

    Returns:
        environment: dictionary with the values of the environment form fields
        geographical_location: dictionary with the values of the geographical location form fields
        other: dictionary with the values of the other form fields
    """

    gpus, cpus, locations = pd.read_csv('datasets/gpus.csv'), pd.read_csv('datasets/cpus.csv'), pd.read_csv('datasets/locations.csv')
    continent = additional_selectboxes['continent_selectbox'].selectbox("Continent", [''] + list(locations['continentName'].unique()))
    source = create_selectbox_with_other(additional_selectboxes['source_selectbox'], additional_other_options['source_other'], df, 'source')
    domain = create_selectbox_with_other(additional_selectboxes['domain_selectbox'], additional_other_options['domain_other'], df, 'domain')
    training_type = create_selectbox_with_other(additional_selectboxes['training_selectbox'], additional_other_options['training_other'], df, 'training_type')
    library_name = create_multiselect_with_other(additional_selectboxes['library_multiselect'], additional_other_options['library_other'], [''] + tags_metadata['libraries'] + ["Other"], 'Libraries')
    region = create_selectbox_with_other(additional_selectboxes['region_selectbox'], additional_other_options['region_other'], locations[locations['continentName']==continent], 'countryName')
    cpu_model = create_selectbox_with_other(additional_selectboxes['cpu_selectbox'], additional_other_options['cpu_other'], cpus, 'model')
    gpu_model = create_selectbox_with_other(additional_selectboxes['gpu_selectbox'], additional_other_options['gpu_other'], gpus, 'model')

    
    environment = {'cpu_model': cpu_model, 'gpu_model': gpu_model}
    geographical_location = {'continent': continent, 'region': region}
    other = {'source': source, 'domain': domain, 'training_type': training_type, 'library_name': library_name}
    return environment, geographical_location, other


def check_attributes(values_dict, warnings_dict):
    """
    Checks if the values of the attributes are valid.
    
    Args:
        values_dict: dictionary with the values of the attributes
        warnings_dict: dictionary with the warning containers

    Returns:
        True if all the attributes are valid, False otherwise
    """

    naming_dict = {'co2_eq_emissions': 'CO2 emissions', 'size': 'model file size', 'datasets_size': 'datasets size', 'downloads': 'downloads', 'accuracy': 'accuracy', 'f1': 'f1', 'other_metrics': 'other metrics', 'num_gpus': 'number of gpus', 'num_cores': 'number of cores', 'ram': 'ram'}
    
    values_to_check = values_dict | values_dict['environment']
    return  all([check_attribute(naming_dict[attribute], values_to_check[attribute], warning_container) for attribute, warning_container in warnings_dict.items()])



def transform_values_by_unit(values_dict, units_dict):
    """
    Transforms the values of the attributes to the units specified in the units dictionary.
    
    Args:
        values_dict: dictionary with the values of the attributes
        units_dict: dictionary with the units of the attributes

    Returns:
        values_dict: dictionary with the values of the attributes transformed to the units specified in the units dictionary
    """
    for unit_attribue, unit_name in units_dict.items():
        values_dict[unit_attribue] = unit_transformation(values_dict[unit_attribue], unit_name)

    values_dict['units'] = units_dict
    return values_dict


def get_size_efficency(values_dict, size_attribute):
    """
    Calculates the efficency of a 'size' attribute of the model.
    
    Args:
        values_dict: dictionary with the values of the attributes
        size_attribute: name of the 'size' attribute of the model

    Returns:
        efficency: the efficency of the 'size' attribute of the model
    """

    co2, size = values_dict['co2_eq_emissions'], values_dict[size_attribute]
    if size is not None and co2 is not None:
        if size == 0 and co2 == 0:
            return None
        else:
            return size/co2 if co2 != 0 else float('inf')
    else:
        return None
    
def preprocess_input_values(values_dict):
    """
    Preprocesses the values of the input attributes.
    
    Args:
        values_dict: dictionary with the values of the attributes

    Returns:
        values_dict: dictionary with the values of the attributes preprocessed
    """

    values_dict['co2_eq_emissions'] = float(values_dict['co2_eq_emissions']) if values_dict['co2_eq_emissions'] != '' else None
    values_dict['size'] = float(values_dict['size']) if values_dict['size'] != '' else None 
    values_dict['datasets_size'] = float(values_dict['datasets_size']) if values_dict['datasets_size'] != '' else None  
    values_dict['downloads'] = int(values_dict['downloads']) if values_dict['downloads'] != '' else None    
    values_dict['accuracy'] = float(values_dict['accuracy']) if values_dict['accuracy'] != '' else None 
    values_dict['f1'] = float(values_dict['f1']) if values_dict['f1'] != '' else None   
    values_dict['datasets'] = values_dict['datasets'].split(',')
    return values_dict
    
def feature_engineer_values(values_dict, df):
    """
    Feature engineers the values of the input attributes.
    
    Args:
        values_dict: dictionary with the values of the attributes
        df: dataframe with the data of the models
    
    Returns:
        values_dict: dictionary with the values of the attributes feature engineered
    """
    values_dict['size_efficency'] = get_size_efficency(values_dict, 'size')
    values_dict['datasets_size_efficency'] = get_size_efficency(values_dict, 'datasets_size')
    values_dict['co2_reported'] = int((values_dict['co2_eq_emissions'] is not None) and values_dict['co2_eq_emissions'] != '')
    other_metrics = string_to_dict(values_dict['other_metrics'])
    values_dict['performance_metrics'] = other_metrics | {'accuracy':values_dict['accuracy'], 'f1': values_dict['f1']}
    values_dict.pop('f1', None); values_dict.pop('accuracy', None)
    values_dict['likes'] = None
    values_dict['performance_score'] = performance_score(values_dict['performance_metrics'], df)
    values_dict['created_at'] = datetime.now().strftime('%Y-%m-%d')
    return values_dict

def reorder_dict(values_dict, df):
    """Reorders the values of the input attributes."""
    return {key: values_dict[key] for key in df.columns}


def prepare_input_for_googlesheet(values_dict):
    """
    Prepares the values of the input attributes to be added to the Google Sheet.
    
    Args:
        values_dict: dictionary with the values of the attributes
    
    Returns:
        values_to_add: list with the values of the attributes to be added to the Google Sheet
    """

    values_dict['geographical_location'] = str(values_dict['geographical_location'])
    values_dict['environment'] = str(values_dict['environment'])
    values_dict['performance_metrics'] = str(values_dict['performance_metrics'])
    values_dict['datasets'] = str(values_dict['datasets'])
    values_dict['units'] = str(values_dict['units'])
    values_dict['library_name'] = str(values_dict['library_name'])

    
    values_to_add = list(values_dict.values())
    return ['' if pd.isnull(value) else value for value in values_to_add]

def create_energy_label(values_dict):
    """
    Creates the energy label of the model.
    
    Args:
        values_dict: dictionary with the values of the attributes
    
    Returns:
        pdf: pdf of the energy label
        png: png of the energy label
    """

    pdf_doc = EnergyLabel(summary=values_dict,  metrics_ref = metrics_ref, boundaries = boundaries)
    pdf_doc.save('energy_label.pdf')
    doc = fitz.open('energy_label.pdf')
    page = doc.load_page(0)  # number of page
    pix = page.get_pixmap()
    output = "energy_label.png"
    pix.save(output)
    doc.close()
    st.markdown('---')
    left_co, cent_co,last_co = st.columns([1,1,5])
    cent_co.image('energy_label.png', width=500)

def handle_form_submission(df, values_dict, warnings_dict, units_dict, to_include):
    """
    Handles the submission of the form.
    
    Args:
        df: dataframe with the data of the models
        values_dict: dictionary with the values of the attributes
        warnings_dict: dictionary with the warnings containers of the attributes
        units_dict: dictionary with the units of the attributes
        to_include: whether to include the model in the dataset or not
    """

    if not check_attributes(values_dict, warnings_dict):
        st.warning('Please fix the warnings before submitting.')
        st.stop()
    
    
    if to_include == 'Yes' and values_dict['modelId'] in df['modelId'].values:
        st.warning('This model already exists in the dataset.')
        st.stop()

    values_dict = transform_values_by_unit(values_dict, units_dict)
    values_dict = preprocess_input_values(values_dict)
    values_dict = feature_engineer_values(values_dict, df)
    values_dict = reorder_dict(values_dict, df)
    create_energy_label(values_dict)

    if to_include == 'Yes':
        values_to_add = prepare_input_for_googlesheet(values_dict)
        sheet.append_row(values_to_add)



def label_form_creation_temp(df):


    with st.form('Label Form'):
        st.markdown("""
                    <h5 style='text-align: center;'>Energy Label Form</h5>
                    <hr style="margin-top: 0;"> 
                    """, unsafe_allow_html=True)
        
        values_dict, warnings_dict, units_dict = create_main_form_fields() 
        additional_values, additional_warnings, additional_selectboxes, additional_other_options = create_additional_form_fields()

        to_include = st.radio(
        "Do you want to include this model to the dataset?",
        ('Yes', 'No'))

        col1, col2, col3 = st.columns([2,1,2])
        submit_button = float(col2.form_submit_button("Generate Label"))

    environment_other_attributes, geographical_location, other = gather_attributes_with_other(additional_selectboxes, additional_other_options)
    warnings_dict |= additional_warnings
    additional_values['environment'] |= environment_other_attributes
    values_dict |= additional_values | other
    values_dict['geographical_location'] = geographical_location

    if submit_button:
        handle_form_submission(df, values_dict, warnings_dict, units_dict, to_include)
    
st.markdown("---")


def dataset_label_creation():
    with st.form('Dataset Label Form'):
        modelId = st.selectbox("Select model from dataset", df_co2['modelId'].unique())
        col1, col2, col3 = st.columns([2,1,2])
        submit_button = float(col2.form_submit_button("Generate Label"))
    if submit_button:
        df_co2_filtered = df_co2[df_co2['modelId'] == modelId].drop_duplicates(subset=['modelId'])
        summary = df_co2_filtered.to_dict('records')[0]
        summary['units'] = {'co2_eq_emissions': 'g', 'size': 'MB', 'datasets_size': 'MB'}
        summary['datasets'] = ast.literal_eval(summary['datasets']) if isinstance(summary['datasets'], str) else []
        create_energy_label(summary)

_, left_column, right_column, _ = st.columns(4)


new_label = left_column.button('Create Label for New Model', on_click=click_unclick, args=('New Model',), disabled=False)
dataset_label = right_column.button('Create Label for Dataset Model', on_click=click_unclick, args=('Datasets',), disabled=False)   

if st.session_state['clicks']['New Model']:
    label_form_creation_temp(df)


if st.session_state['clicks']['Dataset']:
    dataset_label_creation()





    

    
