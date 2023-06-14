import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from energy_label import calculate_compound_rating
from matplotlib.ticker import FuncFormatter

pd.options.mode.chained_assignment = None


import matplotlib

RA = '#639B30'
RB = '#B8AC2B'
RC = '#F8B830'
RD = '#EF7D29'
RE = '#E52421'
COLORS = [RA, RB, RC, RD, RE]

# boundaries = json.load(open('boundaries.json'))

def plot_efficency_distribution(df, width=None, height=None):
    """
    Plot the distribution of energy efficiency labels.

    Args:
    df : dataframe
    width : width of plot
    height : height of plot
    """

    # Count the number of occurrences for each category
    category_counts = df['compound_rating'].value_counts().sort_index()

    # Define color for each category
    colors = {
        'A': 'green',
        'B': '#adff2f',  # Light green-yellow
        'C': 'yellow',
        'D': '#ffbf00',  # Amber
        'E': 'red'
    }

    # Plotting
    fig = plt.figure()  # Set the figure size
    sns.set(style="whitegrid")  # Set the style to 'whitegrid' for better visibility
    sns.barplot(x=category_counts.index, y=category_counts.values, palette=[colors[i] for i in category_counts.index])  # Create a barplot

    # Setting the labels and title for our plot
    plt.xlabel('Energy Labels', fontsize=12)
    plt.ylabel('Count', fontsize=12)


    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()



def plot_model_count(df, width=None, height=None):
    """
    Plot the number of models reporting CO2 emissions over time.

    Args:
    df : dataframe
    width : width of plot
    height : height of plot
    """

    df['year_month'] = pd.to_datetime(df['year_month'])
    grouped_data = df.groupby('year_month').agg({'modelId': 'count'}).reset_index()

    # Convert datetime objects to numerical values for regression
    ref_date = pd.Timestamp('1970-01-01')
    grouped_data['year_month_num'] = (grouped_data['year_month'] - ref_date) / pd.Timedelta(1, 'D')

    # Plot the evolution of the number of models reporting CO2 emissions
    fig = plt.figure()
    ax = sns.lineplot(data=grouped_data, x='year_month', y='modelId', marker='o', label='Data')
    sns.regplot(data=grouped_data, x='year_month_num', y='modelId', scatter=False, label='Linear Regression', ax=ax, line_kws={'alpha': 0.5})  # Add regression line to the plot

    ax.set_ylabel('Number of Models Reporting CO2e', fontsize=11)
    ax.set_xlabel(None)
    plt.xticks(rotation=45, fontsize=10)
    ax.grid(False)
    sns.despine(left=True, bottom=False)
    plt.legend()

    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()



def plot_emissions_reporting_evolution(df, width=None, height=None):
    """
    Plot the evolution of the ratio of models reporting CO2 emissions over the total number of models.

    Args:
    df : dataframe
    width : width of plot
    height : height of plot
    """

    xtick_interval = 3
    df['year_month'] = pd.to_datetime(df['year_month'])
    grouped_data = df.groupby('year_month').agg({'co2_reported': 'sum', 'modelId': 'count'}).reset_index()
    # Calculate the ratio of models reporting CO2 emissions over the total number of models
    grouped_data['co2_reporting_ratio'] = (grouped_data['co2_reported'] / grouped_data['modelId']) * 100

    # Convert datetime objects to numerical values
    ref_date = pd.Timestamp('1970-01-01')
    grouped_data['year_month_num'] = (grouped_data['year_month'] - ref_date) / pd.Timedelta(1, 'D')


    # Plot the evolution of the ratio of models reporting CO2 emissions over the total number of models
    fig=plt.figure()

    ax = sns.lineplot(data=grouped_data, x='year_month', y='co2_reporting_ratio', marker='o', label='Data')
    sns.regplot(data=grouped_data, x='year_month_num', y='co2_reporting_ratio', scatter=False, label='Linear Regression', ax=ax, line_kws={'alpha': 0.5})

    ax.set_ylabel('% of Models Reporting CO2e Emissions', fontsize=11)
    # xticks = grouped_data['year_month'][::xtick_interval]
    # xticks = xticks.dt.strftime('%Y-%m')  # Convert Timestamp objects to strings
    # ax.set_xticks(xticks)
    ax.set_xlabel(None)
    plt.xticks(rotation=45, fontsize=10)
    ax.grid(False)
    sns.despine(left=True, bottom=False)
    plt.legend()

    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_emissions_reported_evolution(df, width=None, height=None):
    """
    Plot the evolution of the average CO2 emissions reported by models over time.

    Args:
    df : dataframe
    width : width of plot
    height : height of plot
    """

    # Group the data by 'year_month' and compute median values for the relevant columns
    grouped_df = df.groupby('year_month').agg({
        'co2_eq_emissions': 'median',
    }).reset_index()

    # Convert 'year_month' to datetime format
    grouped_df['year_month'] = pd.to_datetime(grouped_df['year_month'])

    # Convert datetime objects to numerical values
    ref_date = pd.Timestamp('1970-01-01')
    grouped_df['year_month_num'] = (grouped_df['year_month'] - ref_date) / pd.Timedelta(1, 'D')

    xtick_interval = 3
    xticks = grouped_df['year_month'][::xtick_interval]

    # Plot the evolution of average CO2eq emissions, model size, and performance
    fig, ax = plt.subplots()
    ax.plot(grouped_df['year_month'], grouped_df['co2_eq_emissions'], 'g-o', label='CO2e Emissions')  # Add 'o' as marker
    sns.regplot(data=grouped_df, x='year_month_num', y='co2_eq_emissions', scatter=False, label='Linear Regression', ax=ax, line_kws={'alpha': 0.5})

    ax.set_xlabel(None)
    ax.set_ylabel('CO2e Emissions', fontsize=11)
    ax.set_xticks(xticks)
    plt.xticks(rotation=45, fontsize=10)
    ax.legend(loc='upper left')

    ax.grid(False)
    sns.despine(left=True, bottom=False) 

    # Set y-ticks manually to not include negative values
    plt.gca().set_ylim(bottom=-9)


    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


    
def scatter_models(df, xmetric, ymetric, xlabel, ylabel, boundaries, ind_or_val='index', xlim=None, ylim=None, named_pos=None, named_pos_discard=True, xlog=False, ylog=False, width=None, height=None):
    """
    Scatter plot of models in the space of two metrics.

    Args:
    df : dataframe
    xmetric : metric to plot on the x-axis
    ymetric : metric to plot on the y-axis
    xlabel : label for the x-axis
    ylabel : label for the y-axis
    boundaries : dictionary of boundaries for the metrics
    ind_or_val : whether to plot the index or the value of the metric
    xlim : x-axis limits
    ylim : y-axis limits
    named_pos : list of named positions to highlight
    named_pos_discard : whether to discard models not in the named positions
    xlog : whether to use a logarithmic scale for the x-axis
    ylog : whether to use a logarithmic scale for the y-axis
    width : width of plot
    height : height of plot
    """
    
    fig, ax = plt.subplots(1, 1)

    # If width or height are provided, they're set to the figure
    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)

    # For each boundary in the x metric, a rectangle is drawn on the plot
    # The color of the rectangle is determined by the compound rating of the metrics
    # The same is done for the y metric
    for xi, (x1, x0) in enumerate(boundaries[xmetric]):
        # we define define x1 and x0 ...
        if xi == 0:
            x1 = 100 if xlim is None else xlim[1]
        if xi == len(boundaries[xmetric]) - 1:
            x0 = 0 if xlim is None else xlim[0]
        for yi, (y1, y0) in enumerate(boundaries[ymetric]):
            # we define y1 and y0 ...
            if yi == 0:
                y1 = 100 if xlim is None else ylim[1]
            if yi == len(boundaries[ymetric]) - 1:
                y0 = 0 if xlim is None else ylim[0]
            color = calculate_compound_rating(ratings=[xi, yi], mode='mean', meanings=COLORS)
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, color=color, alpha=.6, zorder=-1))
    xmin, xmax, ymin, ymax = 1e12, 0, 1e12, 0

    # White dashed lines are drawn at x=1 and y=1
    plt.axhline(y=1, color='w', linestyle='--')
    plt.axvline(x=1, color='w', linestyle='--')

    # Lists to store values for the scatter plot
    x, y, n, r = [], [], [], []
    
    # Top 50 models with the most likes for each compound rating are selected
    top_models =  df.groupby('compound_rating').apply(lambda x: x.nlargest(50, 'likes')).reset_index(drop=True)

    # If 'index' is selected, the metric indices are used for the scatter plot, otherwise, the metric values are used
    if ind_or_val == 'index':
        x = top_models[f'{xmetric}_index']
        y = top_models[f'{ymetric}_index']
    else:
        x = top_models[xmetric]
        y = top_models[ymetric]

    # Model names and ratings are stored
    n = top_models['modelId']
    mapping_dict = {'A': COLORS[0], 'B': COLORS[1], 'C': COLORS[2], 'D': COLORS[3], 'E': COLORS[4]}
    r = top_models['compound_rating'].map(mapping_dict)

    # The x and y ranges are updated based on the data
    xmin = min(xmin, min(x))
    xmax = max(xmax, max(x))
    ymin = min(ymin, min(y))
    ymax = max(ymax, max(y))

    # we draw the scatter plot
    ax.scatter(x, y, s=75, marker='o', color=r, edgecolors='white')

    # If no axis limits are provided, they're calculated based on the data
    if xlim is None:
        xlim = [xmin - (xmax-xmin) * 0.05, xmax + (xmax-xmin) * 0.05]
    if ylim is None:
        ylim = [ymin - (ymax-ymin) * 0.05, ymax + (ymax-ymin) * 0.05]

    # Axis labels are set
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Attempt to set the axes scales (linear or logarithmic) and plot the figure
    # If an error occurs (due to a latex error bug from streamlit), 
    # a fallback format is used and the figure is plotted
    try:
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
            
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
            
        plt.tight_layout()
        st.pyplot(fig)
    except:
        if xlog:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:0.0e}'.format(x)))
            ax.tick_params(axis='x', labelsize=9)
        if ylog:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:0.0e}'.format(y)))
            ax.tick_params(axis='y', labelsize=9)
            
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
            
        plt.tight_layout()
        st.pyplot(fig)
        
    plt.close()



    

    

