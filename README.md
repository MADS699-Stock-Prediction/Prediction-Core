# Introduction

Welcome to the Team 17 Moneymakers. Our goal is to predict future returns of individual stocks by analyzing various technical, historic financial KPIs along with sentiments extracted out of social media via natural language processing (NLP). This project will help improve understanding of the machine learning application in the area of stock market prediction.

# Experiment Design (Architecture, Flow and Design)

![architecture diagram](https://github.com/MADS699-Stock-Prediction/Prediction-Core/assets/6002688/3f98052e-7a8c-4546-9e98-8c87c87eed27)

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

5. To evaluate various model metrics (produces result csv file data/model/metrics.csv)
  
   ./run_TSLA_F_GM_TM_all_combination_train_evaluate
   
6. To reproduce all results including data collections,sentiment analysis (require eodhist API key for eodscrapper under src/scrapper folder)

     make
   ./run_TSLA_F_GM_TM_all_combination_train_evaluate
   
# Project Structure
The project follows a standard layout. The major components of the directory structure are:
[under cd Prediction-Core/src  ] 

- src: Source code for the project, including utilities, model training, text processing, and make and evaluation scripts.
- data/raw_data: Directory for data files includes raw technical, fundamental and news data(if EODHIST API is available).
- data/clean_data/all_combined: Directory for clean data files. Combined data file with sentiment for easy research reproducibility.
- data/clean_data/causal_inference: Granger causality for each pair of features including label(log_ret) in this case.
- data/visualization: Contains the visuals used in the report. 
- data_processor : data clean up and processing python code. 
- models: Stores various trained models. It is used during final evaluation of model with unknown stock data.
- model_factory : Python code for model training and evaluation including timeseries, regression, lstm and ann.
- scrapper: Various scrappers for yfiance data, fundamental data, reddit data and eod news related to stock/s.
- proof_of_concept_test: Various raw form of PoC/experiments and data this is a collection of standalone Jupyter notebooks for data extraction to visualizations.
- visual_causal_inference: Source code for the project, including utilities, model training, text processing, and evaluation scripts.
- run_TSLA_F_GM_TM_all_combination_train_evaluate : A shell script to quickly train ANN,LSTM,Regression and TimeSeries model. It also generates model prediction evaluation metrics on unseen data i.e. new stock ticker data.
  
# Statement of Work

Prediction_Core is a team work by Atul Kumar, Kiran Hedge and GyungYoon Park(Nathan).

# Note
To reproduce complete research data and outcome purchase the news API key from [eodhistoricdata](https://eodhd.com/) 
and use the same for eodscrapper under src/scrapper.

# License

GNU General Public License v3.0

