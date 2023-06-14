import streamlit as st
import gspread
import pandas as pd
import ast
import datetime as dt
import numpy as np
from google.oauth2.service_account import Credentials

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']



@st.cache_data(show_spinner=False)
def read_sheet():
    """Reads the Google Sheet and returns a gspread object"""
    creds = Credentials.from_service_account_info(st.secrets['gcp_service_account'], scopes=scope)    
    client = gspread.authorize(creds)
    sheet = client.open("HFStreamlit").sheet1
    return sheet

@st.cache_data(show_spinner=False)
def get_data_from_sheet(_sheet):
    """Reads the Google Sheet and returns a pandas dataframe
    
    Args:
        _sheet (gspread object): gspread object
        
    Returns:
        pandas dataframe: dataframe with the data from the Google Sheet
    """

    data = _sheet.get_all_records(numericise_ignore=['all'])
    df = pd.DataFrame(data)
    return df

@st.cache_data(show_spinner=False)
def cols_preprocess(df):
    """Preprocesses the columns of the dataframeç
    
    Args:
        df (pandas dataframe): dataframe with the data from the Google Sheet
    
    Returns:
        pandas dataframe: dataframe with the data from the Google Sheet preprocessed
    """

    float_cols = ['co2_eq_emissions', 'size_efficency', 'datasets_size_efficency', 'size', 'performance_score', 'datasets_size']
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['co2_eq_emissions'] = df['co2_eq_emissions'].replace('', np.nan).astype(float)
    df['downloads'] = pd.to_numeric(df['downloads'], errors='coerce')
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
    df['co2_reported'] = pd.to_numeric(df['co2_reported'], errors='coerce')
    df[['downloads', 'likes', 'co2_reported']] = df[['downloads', 'likes', 'co2_reported']].astype('Int64')
    df['library_name'] = df['library_name'].apply(lambda libraries:  ast.literal_eval(libraries) if not isinstance(libraries, list) else libraries)
    df = df.replace({pd.NA: np.nan})
    return df



@st.cache_data(show_spinner=False)
def read_data(_sheet=None):
    """
    Reads the Google Sheet and returns a pandas dataframe with the data preprocessed
    
    Args:
        _sheet (gspread object, optional): gspread object. Defaults to None.

    Returns:
        pandas dataframe: dataframe with the data from the Google Sheet preprocessed
    """

    if _sheet is None:
        _sheet = read_sheet()
    df = get_data_from_sheet(_sheet)
    df = cols_preprocess(df)
    return df