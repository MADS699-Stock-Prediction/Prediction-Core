# Introduction

Welcome to the Team 17 Moneymakers. Our goal is to predict future returns of individual stocks by analyzing various technical, historic financial KPIs along with sentiments extracted out of social media via natural language processing (NLP). This project will help improve understanding of the machine learning application in the area of stock market prediction.

# Getting Started

Follow the steps outlined below to set up and run the project:

# Prerequisites

Python 3.7 or higher - the primary programming language

# Reproducing this research

1. Clone the repository:

    git clone https://github.com/MADS699-Stock-Prediction/Prediction-Core.git

2. Navigate to the project directory:
   
    cd Prediction-Core/src

4. install project dependencies

    pip install -r requirements.txt

5. make

# Project Structure
The project follows a standard layout. The major components of the directory structure are:
[under cd Prediction-Core/src  ] 

- data: Directory for data files, processed data, and model outputs.
- report_visuals: Contains the visuals used in the report. 
- data_processor : data clean up and processing. 
- models: Stores the various trained models.
- model_factory : various models including timeseries, regression, lstm and ann.
- scrapper: Various scrappers for yfiance data, fundamental data, reddit data and various news related to stock/s.
- notebooks: Contains Jupyter notebooks for data preparation, model training, and evaluation.
- proof_of_concept_test: Various raw form of PoC this is a collection of standalone Jypyter noteboos for data extraction to visualizations.
- src: Source code for the project, including utilities, model training, text processing, and evaluation scripts.

# Statement of Work
Prediction_Core is a team work by Atul Kumar, Kiran Hedge and GyungYoon Park(Nathan).

# License

GNU General Public License v3.0

