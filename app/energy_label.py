import pandas as pd
import numpy as np
import streamlit as st
import ast

# Weightage of different metrics for energy efficiency computation
METRIC_WEIGHTS = {
    'co2_eq_emissions': 0.35,
    'size_efficency': 0.1,
    'datasets_size_efficency': 0.1,
    'downloads': 0.25,
    'performance_score': 0.2
}

# Metrics where a higher value is better
HIGHER_BETTER = [
    'performance_score',
    'size_efficency',
    'datasets_size_efficency',
    'downloads'
]

# Metrics to be scaled by a power transformation
POWER_SCALE_METRICS = [
    'downloads',
    'co2_eq_emissions'
]

@st.cache_data
def read_df_processed():
    """Reads processed dataset from CSV and applies necessary transformations."""
    df = pd.read_csv('datasets/HFCO2.csv')
    
    # Convert string representation of library list to actual list
    df['library_name'] = df['library_name'].apply(lambda libraries:  ast.literal_eval(libraries) if not isinstance(libraries, list) else libraries)
    
    return df


# Turn off pandas warning for chained assignment
pd.options.mode.chained_assignment = None

def weighted_mean(ratings, weights):
    """
    Compute weighted mean of ratings.

    Args:
    ratings : list of ratings
    weights : corresponding weights for the ratings

    Returns:
    weighted mean : weighted mean of ratings
    """
    # Remove NaN values and their corresponding weights
    not_nan_indices = np.isfinite(ratings)
    ratings_clean = np.array(ratings)[not_nan_indices]
    weights_clean = np.array(weights)[not_nan_indices]
    
    # Renormalize the weights
    weights_clean = weights_clean / np.sum(weights_clean)

    # Compute the weighted mean
    mean = np.sum(ratings_clean * weights_clean)
    
    return int(round(mean))

def assign_rating(index, boundaries):
    """
    Assigns a rating based on index value. If index is within certain boundaries, 
    a specific rating is given.

    Args:
    index : index value for which rating is to be assigned
    boundaries : list of boundaries for rating

    Returns:
    rating : assigned rating
    """
    if index is None or pd.isnull(index):
        return np.nan

    for i, (upper, lower) in enumerate(boundaries):
        if index <= upper and index > lower:
            return i
    return 4 # worst rating if index does not fall in boundaries

def calculate_compound_rating(ratings, weights=None, meanings='ABCDE', mode='mean'):
    """
    Calculates a compound rating based on individual ratings and their weights.

    Args:
    ratings : list of individual ratings
    weights : list of weights corresponding to the ratings
    meanings : string representing different ratings, default is 'ABCDE'
    mode : method to calculate compound rating, default is 'mean'

    Returns:
    compound rating
    """
    if all(x is None or pd.isna(x) for x in ratings):
        return None
    if weights is None:
        weights = [1.0 / len(ratings) for _ in ratings]
    if mode == 'mean':
        return meanings[weighted_mean(ratings, weights)]

def value_to_index(value, ref, metric):
    """
    Convert a value to an index by normalizing it with a reference value.
    If the metric is higher better, index is value divided by reference value,
    otherwise index is reference value divided by value.

    Args:
    value : value to be converted
    ref : reference value for normalization
    metric : name of the metric

    Returns:
    index
    """
    if pd.isnull(value) or value is None:
        return None

    try:
        return value / ref if metric in HIGHER_BETTER else ref / value
    except:
        return 0

@st.cache_data
def compute_boundaries(df, metric, metrics_ref, index=True):
    """
    Compute the boundaries for assigning ratings based on the distribution of metric values.
    Args:
    df : DataFrame containing the metrics
    metric : the metric for which to compute the boundaries
    metrics_ref : reference metrics for normalization
    index : if True, convert metric values to indices before computing boundaries
    Returns:
    boundaries for ratings
    """
    if index:
        metric_parameters = df[metric].apply(lambda x: value_to_index(x, metrics_ref[metric], metric))
    else:
        metric_parameters = df[metric]

    if metric == 'co2_eq_emissions':
        return np.nanpercentile(np.sqrt(metric_parameters), [20, 40, 60, 80])[::-1]
    if metric == 'downloads':
        return np.nanpercentile(metric_parameters, [60,85,94,96])[::-1]
    else:
        return np.nanpercentile(metric_parameters, [20, 40, 60, 80])[::-1]



@st.cache_data
def load_ref_metrics(df, model_name='distilgpt2'):
    """
    Load reference metrics for a specified model from the dataset. If the value is missing,
    the median value for the metric in the dataset is used.
    Args:
    df : DataFrame containing the metrics
    model_name : name of the model to use as reference
    Returns:
    dictionary of reference metrics
    """
    metrics = list(METRIC_WEIGHTS.keys())
    ref_model = df[df['modelId'] == model_name].iloc[0]

    metrics_ref = {metric: ref_model[metric] if not pd.isnull(ref_model[metric]) else df[metric].median() for metric in metrics}

    return metrics_ref


def load_boundaries(df, metrics, metrics_ref, index=True):
    """
    Compute boundaries for all metrics and convert them to intervals.
    Args:
    df : DataFrame containing the metrics
    metrics : list of metrics for which to compute boundaries
    metrics_ref : reference metrics for normalization
    index : if True, convert metric values to indices before computing boundaries
    Returns:
    dictionary of boundary intervals for each metric
    """
    boundaries = {metric:compute_boundaries(df, metric, metrics_ref) for metric in metrics}

    max_value = float('inf')
    min_value = 0

    boundary_intervals = {}

    for key, boundaries in boundaries.items():
        intervals = [[max_value, boundaries[0]]]
        for i in range(len(boundaries)-1):
            intervals.append([boundaries[i], boundaries[i+1]])
        intervals.append([boundaries[-1], min_value])
        
        boundary_intervals[key] = intervals

    return boundary_intervals

def assign_energy_label(metrics, metrics_ref, boundaries, meanings, rating_mode, index=True):
    """
    Assigns an energy efficiency label based on the metrics.

    Args:
    metrics : dictionary of metrics
    metrics_ref : reference metrics for normalization
    boundaries : boundary intervals for each metric
    meanings : string representing different ratings
    rating_mode : method to calculate compound rating
    index : if True, convert metric values to indices before assigning label

    Returns:
    compound rating and dictionary of individual ratings
    """
    weights = list(METRIC_WEIGHTS.values())
    if index:
        metrics = {metric: value_to_index(value, metrics_ref[metric], metric) for metric, value in metrics.items()}

    metrics_to_rating = (
        {metric:
          assign_rating(value, boundaries[metric])
          for metric, value in metrics.items()})
    
    ratings = list(metrics_to_rating.values())
    return calculate_compound_rating(ratings, weights, meanings, rating_mode), metrics_to_rating




def add_index_metrics(df):
    """
    Add index metrics to dataframe.

    Args:
    df : dataframe

    Returns:
    dataframe with index metrics
    """
    metrics = list(METRIC_WEIGHTS.keys())
    metrics_ref = load_ref_metrics(df)
    for metric in metrics:
        df[metric + '_index'] = df[metric].apply(lambda value: value_to_index(value, metrics_ref[metric], metric))
    return df
    

def wrapper_assign_energy_label(row, metrics_ref, boundaries, meanings, rating_mode, index=True):
    """
    Wrapper function for assign_energy_label to be used with pandas apply.
    """

    val1, val2 = assign_energy_label(row, metrics_ref, boundaries, 'ABCDE', 'mean') 
    return pd.Series([val1, val2])
    
@st.cache_data
def assign_energy_label_to_df(df, index=True):
    """
    Assigns energy efficiency labels to dataframe.

    Args:
    df : dataframe
    index : if True, convert metric values to indices before assigning label

    Returns:
    dataframe with energy efficiency labels
    """
                       
    metrics = list(METRIC_WEIGHTS.keys())
    metrics_ref = load_ref_metrics(df)
    
    boundaries = load_boundaries(df, metrics, metrics_ref)


    df[['compound_rating', 'metrics_to_rating']] = df[metrics].apply(lambda row: wrapper_assign_energy_label(row, metrics_ref, boundaries, 'ABCDE', 'mean') , axis=1)
    

    return df
