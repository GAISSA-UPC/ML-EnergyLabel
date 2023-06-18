import pandas as pd
import streamlit as st
import ast
from itertools import chain
import datetime as dt
import plots
import energy_label
import time
from data import read_data
import numpy as np


st.set_page_config(page_title='Hugging Face Dashboard',
                   page_icon=':bar_chart:',
                   layout='wide')


# Based on https://github.com/GreenAlgorithms/green-algorithms-tool reference values
def emissions_to_equivalence(co2):
    """Converts the emissions to a flight equivalence"""
    ref_df = pd.read_csv('../datasets/referenceValues.csv', header=1)
    flight_PAR_LON = ref_df[ref_df['variable'] == 'flight_PAR-LON']['value'].values[0]
    flight_NY_SF = ref_df[ref_df['variable'] == 'flight_NY-SF']['value'].values[0]
    flight_NY_MEL = ref_df[ref_df['variable'] == 'flight_NYC-MEL']['value'].values[0]
    if co2 < 0.5 * flight_NY_SF:
            return f'Equivalent to {round(co2 / flight_PAR_LON * 100)} % of a flight from Paris to London'
    elif co2 < 0.5 * flight_NY_MEL:
        return f'Equivalent to {round(co2 / flight_NY_SF * 100)} % of a flight from NYC to San Francisco'
    else:
        return f'Equivalent to {round(co2 / flight_NY_MEL * 100)} % of a flight from NYC to Melbourne'

# we read the data from google sheet
with st.spinner('Loading data...'):
    df = read_data()


# we read carbon emissions data and assign energy labels
df_co2 = df[df['co2_reported'] == True]
df_co2 = energy_label.assign_energy_label_to_df(df_co2)
df_co2 = energy_label.add_index_metrics(df_co2)

# we load the metrics reference and the boundaries for future use
metrics_ref = energy_label.load_ref_metrics(df_co2)
boundaries = energy_label.load_boundaries(df_co2, list(energy_label.METRIC_WEIGHTS.keys()), metrics_ref)

# basic preprocessing
df_co2 = df_co2.drop_duplicates(subset='modelId', keep='first')
df_co2 = df_co2[df_co2['co2_eq_emissions'] > 0]


# --- SIDEBAR --- filtering options
    

st.sidebar.header("Apply the Filters: ")


st.sidebar.markdown('Select the carbon emissions range (CO2e):')
left_column, right_column = st.sidebar.columns(2)
min_co2 = left_column.number_input('Min CO2e', step=0.5)
max_co2 = right_column.number_input('Max CO2e', value=df_co2['co2_eq_emissions'].max(), step=0.5)

st.sidebar.markdown('Select the downloads range:')
left_column, right_column = st.sidebar.columns(2)
min_downloads = left_column.number_input('Min downloads', step=1)
max_downloads = right_column.number_input('Max downloads', value=df_co2['downloads'].max(), step=1)

st.sidebar.markdown('Select the likes range:')
left_column, right_column = st.sidebar.columns(2)
min_likes = left_column.number_input('Min likes', step=1)
max_likes = right_column.number_input('Max likes', value=df_co2['likes'].max(), step=1)


format = 'MMM DD, YYYY'  # format output
start_date = dt.date(year=2021,month=10,day=1)
end_date = dt.date(year=2023,month=3,day=31)
max_days = end_date-start_date
min_date, max_date = st.sidebar.slider('Select date range', min_value=start_date, value=(start_date, end_date) , max_value=end_date, format=format)


domain = st.sidebar.multiselect(
    "Select the ML application domain",
    options=df_co2['domain'].unique(),
    default=df_co2['domain'].unique()
)

if not domain:
    st.warning('Please select at least one domain')
    st.stop()

training_type = st.sidebar.multiselect(
    "Select the training type",
    options=df_co2['training_type'].unique(),
    default=df_co2['training_type'].unique()
)

if not training_type:
    st.warning('Please select at least one training type')
    st.stop()

library_name = st.sidebar.multiselect(
    "Select the library name",
    options=list(set(chain.from_iterable(df_co2['library_name']))),
    default=list(set(chain.from_iterable(df_co2['library_name'])))
)

if not library_name:
    st.warning('Please select at least one library')
    st.stop()

if not library_name:
    st.warning('Please select at least one library')
    st.stop()

emissions_source = st.sidebar.multiselect(
    "Select the emissions source",
    options=df_co2['source'].unique(),
    default=df_co2['source'].unique()
)

if not emissions_source:
    st.warning('Please select at least one emissions source')
    st.stop()


# datetime preprocessing
df_co2['created_at'] = pd.to_datetime(df_co2['created_at'], errors='coerce')
df_co2['year_month'] = df_co2['created_at'].apply(lambda x: x.strftime('%Y-%m') if not pd.isnull(x) and pd.notna(x) else np.nan)


# we apply the selected filters
df_co2_filtered = df_co2[
    (df_co2['co2_eq_emissions'] >= min_co2) &
    (df_co2['co2_eq_emissions'] <= max_co2) &
    (df_co2['downloads'] >= min_downloads) &
    (df_co2['downloads'] <= max_downloads) &
    (df_co2['likes'] >= min_likes) &
    (df_co2['likes'] <= max_likes) &
    (df_co2['created_at'].dt.date >= min_date) &
    (df_co2['created_at'].dt.date <= max_date) &
    (df_co2['domain'].isin(domain)) &
    (df_co2['training_type'].isin(training_type)) &
    (df_co2['source'].isin(emissions_source))
]
df_co2_filtered = df_co2_filtered[df_co2_filtered['library_name'].apply(lambda x: any([lib in x for lib in library_name]))]


if df_co2_filtered.empty:
    st.warning('No models found with the selected filters.')
    st.stop()

# --- END SIDEBAR ---

# --- MAIN PAGE ---

st.title(':seedling: :bar_chart: Hugging Face Emissions Dashboard')


# basic statistics
total_carbon_emissions = df_co2_filtered['co2_eq_emissions'].sum()
average_carbon_emissions = df_co2_filtered['co2_eq_emissions'].mean()
left_column, middle_column, right_column = st.columns(3)
rating_to_star_mapping = {'A': 5, 'B': 4, 'C': 3, 'D':2, 'E':1}
with left_column:
    st.subheader("Total Emissions:")
    st.subheader(f"CO2e {round(total_carbon_emissions/1000,2):,} Kg", help = emissions_to_equivalence(total_carbon_emissions))

with middle_column:
    st.subheader("Average Emissions per Model:")
    st.subheader(f"CO2e {round(average_carbon_emissions/1000,2):,} Kg", help = emissions_to_equivalence(average_carbon_emissions))

with right_column:
    st.subheader("Most Common Effficency Rating:")
    most_common_rating = rating_to_star_mapping[df_co2_filtered['compound_rating'].value_counts().idxmax()]
    st.subheader(':star:'*most_common_rating + '★'*(5-most_common_rating))



st.markdown("---")

left_column, right_column = st.columns(2)

# we make the visualizations of the main page
with left_column:
    st.subheader("Number of Models Reporting CO2e each Month")
    plots.plot_model_count(df_co2_filtered, width=5, height=4)
    st.subheader("Distribution of Carbon Efficiency Labels")
    plots.plot_efficency_distribution(df_co2_filtered, width=5, height=4)
    st.subheader("CO2e vs Downloads Tradeoff")
    downloads_index_max = df_co2_filtered['downloads_index'].max() + 0.1*df_co2_filtered['downloads_index'].max()
    downloads_index_min = df_co2_filtered['downloads_index'].min() - 0.1*df_co2_filtered['downloads_index'].min()

    plots.scatter_models(df_co2_filtered, xmetric='co2_eq_emissions', ymetric='downloads', xlabel='CO2e (log index scale)', ylabel='downloads (index scale)',
                          xlim=(df_co2_filtered['co2_eq_emissions_index'].min(), df_co2_filtered['co2_eq_emissions_index'].max()),
                            ylim=(downloads_index_min, downloads_index_max), boundaries=boundaries, named_pos=[], xlog=True, ylog=True, width=5, height=4)
with right_column:
    st.subheader("Median Model Carbon Emissions per Month")
    plots.plot_emissions_reported_evolution(df_co2_filtered, width=5, height=4)
    st.subheader("CO2e vs Performance Score Tradeoff")
    performance_score_index_max = df_co2_filtered['performance_score_index'].max() + 0.1*df_co2_filtered['performance_score_index'].max()
    performance_score_index_min = df_co2_filtered['performance_score_index'].min() - 0.1*df_co2_filtered['performance_score_index'].min() 
    plots.scatter_models(df_co2_filtered, xmetric='co2_eq_emissions', ymetric='performance_score', xlabel='CO2e (log index scale)', ylabel='Performance Score (index scale)',
                          xlim=(df_co2_filtered['co2_eq_emissions_index'].min(), df_co2_filtered['co2_eq_emissions_index'].max()),
                          ylim=(performance_score_index_min,performance_score_index_max), boundaries=boundaries, named_pos=[], xlog=True, width=5, height=4)
    st.subheader("CO2e vs Model Size Efficency Tradeoff")
    size_efficency_index_max = df_co2_filtered['size_efficency_index'].max() + 0.1*df_co2_filtered['size_efficency_index'].max()
    size_efficency_index_min = df_co2_filtered['size_efficency_index'].min() - 0.1*df_co2_filtered['size_efficency_index'].min() 
    plots.scatter_models(df_co2_filtered, xmetric='co2_eq_emissions', ymetric='size_efficency', xlabel='CO2e (log index scale)', ylabel='Size Efficency (index scale)',
                          xlim=(df_co2_filtered['co2_eq_emissions_index'].min(), df_co2_filtered['co2_eq_emissions_index'].max()),
                          ylim=(size_efficency_index_min,size_efficency_index_max), boundaries=boundaries, named_pos=[], xlog=True, ylog=True, width=5, height=4)

