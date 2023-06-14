import streamlit as st
import webbrowser
import energy_label
import pandas as pd
import ast
import json
from streamlit_extras.switch_page_button import switch_page
from scipy import stats
import fitz
import os
from label_generation import EnergyLabel
from energy_label import compute_boundaries



# if os.path.exists('energy_label.png'):
#     os.remove('energy_label.png')

def standardize_name(name: str) -> str:
    return name.lower().replace("_", " ")


st.set_page_config(
    page_title="Efficency Label Generation",
    page_icon=":bar_chart:",
)

st.title("Efficiency Label Generator")


@st.cache_data
def read_df_processed():
    df = pd.read_csv('datasets/HFStreamlit.csv')
    df['library_name'] = df['library_name'].apply(lambda libraries:  ast.literal_eval(libraries) if not isinstance(libraries, list) else libraries)
    # df['datasets'] = df['datasets'].apply(lambda datasets: [''] if pd.isnull(datasets) else [datasets] if '[' not in datasets else ast.literal_eval(datasets))

    return df


df = read_df_processed()
ref_model = df[df['modelId'] == 'distilgpt2'].iloc[0]

boundaries = json.load(open('boundaries.json'))

metrics = list(boundaries.keys())

metrics_ref = {metric: ref_model[metric] if ref_model[metric] is None else df[metric].median()
              for metric in metrics}


st.markdown(
    """
    Here you have the opportunity to gauge the energy efficiency of a model. 
    By entering details about a specific machine learning model, our system will
    generate an energy label ranging from A (indicating high energy efficiency)
    to E (indicating poor energy efficiency). The labels are determined based on several factors including CO2 emissions,
    model size, reusability, and overall performance. This tool is a useful
    guide for AI practitioners seeking to develop and promote more sustainable
    models.

    **If your model does not have some attribute, leave it as blank**
    """
)

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def performance_score(df):
    metrics = ['accuracy', 'f1', 'rouge1', 'rougeL']
    
    df['f1'] = min_max_normalize(df['f1'])
    df['accuracy'] = min_max_normalize(df['accuracy'])
    df['rouge1'] = min_max_normalize(df['rouge1'])
    df['rougeL'] = min_max_normalize(df['rougeL'])
    
    return df.apply(lambda row: stats.hmean([row[metric] for metric in metrics if not np.isnan(row[metric])]), axis=1)

def performance_str_to_score(metrics_dict, df):

    
    normalized_metrics = [(value - df[metric].min()) / (df[metric].max() - df[metric].min()) for metric,value in metrics_dict.items()]
    return stats.hmean(normalized_metrics)


with st.form('Label Form'):
    model_name = st.text_input("Model name")
    dataset = st.text_input("Dataset")
    environment = st.text_input("Environment")
    co2 = st.text_input("CO2 emissions (CO2eq g)")
    source = st.text_input("Emissions Source")
    model_file_size = st.text_input("Model File Size (Bytes)")
    datasets_size = st.text_input("Dataset Size (Bytes)")
    accuracy = st.text_input("Accuracy")
    f1 = st.text_input("F1")
    downloads = st.text_input("Downloads")
    submit_button = float(st.form_submit_button("Generate Label"))

    


if submit_button:
    co2 = float(co2)
    model_file_size = float(model_file_size)
    datasets_size = float(datasets_size)
    downloads = float(downloads)
    size_efficency = model_file_size/co2
    accuracy, f1 = float(accuracy), float(f1)
    performance_score = performance_str_to_score({'accuracy':accuracy, 'f1':f1}, df)
    model_summary = {'modelId': model_name, 'dataset':dataset, 'co2_eq_emissions': co2, 'source': source, 'size_efficency': size_efficency, 'size': model_file_size, 'accuracy': accuracy, 'f1': f1, 'performance_score': performance_score, 'downloads': downloads}
    
    
    pdf_doc = EnergyLabel(summary=model_summary, metrics_ref=metrics_ref, boundaries=boundaries, rating_mode='mean')
    

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
