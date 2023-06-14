import pandas as pd
import streamlit as st
import ast
from itertools import chain
from dateutil.relativedelta import relativedelta 
import datetime as dt
import plots
import energy_label


st.set_page_config(page_title='Hugging Face Dashboard',
                   page_icon=':bar_chart:',
                   layout='wide')


@st.cache_data
def read_df_processed():
    df = pd.read_csv('datasets/HFStreamlit.csv')
    df['library_name'] = df['library_name'].apply(lambda libraries:  ast.literal_eval(libraries) if not isinstance(libraries, list) else libraries)

    return df


df = read_df_processed()
df_co2 = df[df['co2_reported'] == True]
ref_model = df_co2[df_co2['modelId'] == 'distilgpt2'].iloc[0]
df_co2 = energy_label.assign_energy_label_to_df(df_co2, ref_model)


# df = df[df['co2_reported']==True]

# --- SIDEBAR ---


st.sidebar.header("Apply the Filters: ")


st.sidebar.markdown('Select the carbon emissdions range (CO2e):')
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


## Range selector
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

training_type = st.sidebar.multiselect(
    "Select the training type",
    options=df_co2['training_type'].unique(),
    default=df_co2['training_type'].unique()
)

library_name = st.sidebar.multiselect(
    "Select the library name",
    options=list(set(chain.from_iterable(df_co2['library_name']))),
    default=list(set(chain.from_iterable(df_co2['library_name'])))
)

emissions_source = st.sidebar.multiselect(
    "Select the emissions source",
    options=df_co2['source'].unique(),
    default=df_co2['source'].unique()
)


st.sidebar.markdown('Select if AutoTrained models')
auto = st.sidebar.checkbox('AutoTrained')

# Convert 'created_at' to datetime
df['created_at'] = pd.to_datetime(df['created_at'])
df_co2['created_at'] = pd.to_datetime(df_co2['created_at'])

# df_co2 = scripts.assign_energy_labels(df_co2, ref_model)

# Apply the filters
df_filtered = df[
    (df['downloads'] >= min_downloads) &
    (df['downloads'] <= max_downloads) &
    (df['likes'] >= min_likes) &
    (df['likes'] <= max_likes) &
    (df['created_at'].dt.date >= min_date) & 
    (df['created_at'].dt.date <= max_date) &
    (df['domain'].isin(domain)) &
    (df['training_type'].isin(training_type)) &
    (df['source'].isin(emissions_source)) 
    # (df['auto'] == auto)
]



df_filtered = df_filtered[df_filtered['library_name'].apply(lambda x: any([lib in x for lib in library_name]))]


df_co2_filtered = df_co2[
    (df_filtered['co2_eq_emissions'] >= min_co2) & 
    (df_filtered['co2_eq_emissions'] <= max_co2) &
    (df['downloads'] >= min_downloads) &
    (df['downloads'] <= max_downloads) &
    (df['likes'] >= min_likes) &
    (df['likes'] <= max_likes) &
    (df['created_at'].dt.date >= min_date) & 
    (df['created_at'].dt.date <= max_date) &
    (df['domain'].isin(domain)) &
    (df['training_type'].isin(training_type)) &
    (df['source'].isin(emissions_source)) 
    # (df['auto'] == auto)
]


df_co2_filtered = df_co2_filtered[df_co2_filtered['library_name'].apply(lambda x: any([lib in x for lib in library_name]))]






# --- MAIN PAGE ---
st.title(':seedling: :bar_chart: Hugging Face Emissions Dashboard')



total_carbon_emissions = round(df_co2_filtered['co2_eq_emissions'].sum(),2)
average_carbon_emissions = round(df_co2_filtered['co2_eq_emissions'].mean(),2)
left_column, middle_column, right_column = st.columns(3)
rating_to_star_mapping = {'A': 5, 'B': 4, 'C': 3, 'D':2, 'E':1}


with left_column:
    st.subheader("Total Emissions:")
    st.subheader(f"CO2e {total_carbon_emissions}")

with middle_column:
    st.subheader("Average Emissions:")
    st.subheader(f"CO2e {average_carbon_emissions}")

with right_column:
    st.subheader("Efficency Rating:")
    st.subheader(':star:'*rating_to_star_mapping[df_co2_filtered['compound_rating'].value_counts().idxmax()])



st.markdown("---")

left_column, right_column = st.columns(2)


with left_column:
    st.subheader("Carbon Emissions Reporting Evolution")
    plots.plot_emissions_reporting_evolution(df_filtered)
    st.subheader("Distribution of carbon efficency labels")
    plots.plot_efficency_distribution(df_co2_filtered)
with right_column:
    st.subheader("Carbon Emissions Reported Evolution")
    plots.plot_emissions_reported_evolution(df_co2_filtered)
    st.subheader("Other plots (e.g the distributions of the models alongside the co2_eq_em - efficency labels plot)")
    st.image('figures/plot_example.png', width=400)
    # scatter_models('co2_eq_emissions', 'performance_score',
    # 'CO2e',
    # 'Performance Score',
    # 'scatter_power_acc.pdf', xlim=(0.1, 2.05), ylim=(0.77, 1.15), named_pos=not_name, width=5, height=5)



# st.dataframe(df_filtered)