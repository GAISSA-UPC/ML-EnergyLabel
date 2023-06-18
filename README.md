# Carbon Footprint of ML Models - A Repository Mining Study and Application

## Overview

This repository provides tools and analysis for understanding and evaluating the carbon footprint of machine learning models, primarily focusing on models from Hugging Face. The project is divided into two main parts: an initial data analysis and a subsequent web-application.

The data analysis seeks to answer two main research questions:
1. How do ML model creators measure and report carbon emissions on Hugging Face?
2. What aspects impact the carbon emissions of training ML models?

The web-application is a user-friendly tool that allows users to estimate the energy efficiency of their models, visualize carbon emissions data from Hugging Face models, and add their own models to the dataset.

You can access the deployed app at [energy-label.streamlit.app](https://energy-label.streamlit.app).

## Repository Structure

- `code/`: Contains the Jupyter notebooks for data extraction, preprocessing, and analysis.
- `app/`: Contains the Streamlit web-application.
- `datasets/`: Contains the raw, processed, and manually curated datasets used for the analysis.
- `metadata/`: Contains the `tags_metadata.yaml` file used during preprocessing.
- `requirements.txt`: Lists the required Python packages to set up the environment and run the code.

### App Folder Structure

- `Home.py`: The homepage of the web-application.
- `pages/`: Contains individual page scripts for the web-application.
    - `1_Efficency_Label.py`: The energy label generation page.
    - `2_Data_Visualization.py`: The data visualization page.
- `energy_label.py`: Script for generating energy labels.
- `label_generation.py`: Script for creating the image/pdf of the energy labels.
- `plots.py`: Contains the plots for the data visualization page.
- `data.py`: Contains functions to read data from Google Sheets.
- `HFStreamlitPreprocessing.ipynb`: Jupyter notebook to apply necessary transformations on HFTotalProcessed.csv for the Streamlit app.
- `label_design/parts`: Contains necessary images for the creation of the energy labels.

## Setup and Execution

1. Set up a Python virtual environment (optional, but recommended). We used Python 3.10.11 for this project.
2. Install the required Python packages using `pip install -r requirements.txt`. Don't forget to install Streamlit with `pip install streamlit` if you want to run the web app locally.
3. If you're planning to use the data analysis part, you need to handle the datasets. We use DVC to manage large datasets.
    1. Install DVC, `pip install dvc`.
    2. Set up the DVC remote storage following the instructions on the [DVC remote storage page](https://dvc.org/doc/command-reference/remote).
    3. Pull the data from the remote storage with `dvc pull`.
4. For the data analysis part, open the Jupyter notebooks in the `code/` folder and follow the instructions in each notebook.
5. To run the web-application locally, navigate to the `app/` folder and run `streamlit run Home.py`.

Remember to cite the original project when using this code for your own research!

## Datasets

This project uses several datasets, which are managed with DVC due to their size. These datasets can be found in the `datasets/` directory after running `dvc pull`.

