import streamlit as st
from streamlit_extras.switch_page_button import switch_page


def standardize_name(name: str) -> str:
        return name.lower().replace("_", " ")


st.set_page_config(
    page_title="Hugging Face Emissions Dashboard",
    page_icon=":bar_chart:",
)


import streamlit as st
st.markdown("<h1 style='text-align: center;'>Welcome to the Green AI Dashboard!</h1>", unsafe_allow_html=True)



import streamlit as st




left_column, right_column = st.columns([4,1])



st.markdown(
    """

    Our application offers two main features:

    1. **Interactive Dashboard**: This offers an in-depth, dynamic view of carbon
    emissions data from the Hugging Face Hub. You can explore various AI model
    attributes like energy consumption and understand their impact on sustainability.

    2. **Energy Efficiency Label Generator**: This tool allows you to evaluate a model's
    energy efficiency. Enter details about an AI model, and our system will provide an
    energy label from A (highly efficient) to E (less efficient), considering factors like
    CO2 emissions, model size, and performance. It assists AI professionals in building
    more sustainable models.
    """
, unsafe_allow_html=True)

st.markdown('----')

_, left_column, right_column, _ = st.columns(4)


with left_column:
    if st.button('Generate Efficiency Label'):
        switch_page('efficiency label')
with right_column:
    if st.button('Open Hugging Face Visualization'):
        switch_page('data visualization')

