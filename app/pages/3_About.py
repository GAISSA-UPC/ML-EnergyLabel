import streamlit as st
from streamlit_extras.switch_page_button import switch_page


def standardize_name(name: str) -> str:
        return name.lower().replace("_", " ")


st.set_page_config(
    page_title="Hugging Face Emissions Dashboard",
    page_icon=":bar_chart:",
)


st.markdown("<h1 style='text-align: center;'>About</h1>", unsafe_allow_html=True)

left_column, right_column = st.columns([4,1])


with left_column:
    st.markdown(
    """

    This interactive web application, developed as part of the [GAISSA](https://gaissa.upc.edu/en) project
    at the [GESSI](https://gessi.upc.edu/en) research group, is designed to provide you with an in-depth
    look at the carbon emissions data for various machine learning models
    hosted on the Hugging Face Hub, and to promote sustainability within
    the AI community.
    """
    )
with right_column:
    st.image('app/figures/gaissa_logo.png')


st.markdown(
    """

    
    By using our platform, you are joining a community of researchers and
    developers committed to addressing the environmental impact of AI.
    Your actions can help drive the development of more sustainable practices
    in machine learning, contributing towards a greener future. Let's embark
    on this journey towards Green AI together!
    """
, unsafe_allow_html=True)


