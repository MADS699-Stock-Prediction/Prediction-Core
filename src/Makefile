#Raw Data => Clean Data => Combined Data => Model

all: models/time_series_model.pkl \
	models/pycaret_model.pkl models/pycaret_regression_model.pkl \
	models/ann.pkl models/lstm.pkl

models/lstm.pkl: process_raw_data model_factory/lstm.py
	python model_factory/lstm.py --stock_ids TSLA \
	-clean_tech_data_store_dir data/clean_data/yfinancetechdata -start_date 2018-06-20 -end_date 2023-06-20 \
	-model_storage_path models/

models/ann.pkl: process_raw_data model_factory/ann.py
	python model_factory/ann.py --stock_ids TSLA \
	-clean_tech_data_store_dir data/clean_data/yfinancetechdata -start_date 2018-06-20 -end_date 2023-06-20 \
	-model_storage_path models/

models/pycaret_regression_model.pkl: process_raw_data model_factory/pycaret_regression_model.py
	python model_factory/pycaret_regression_model.py --stock_ids TSLA \
	-clean_tech_data_store_dir data/clean_data/yfinancetechdata -start_date 2018-06-20 -end_date 2023-06-20 \
	-model_storage_path models/


models/pycaret_model.pkl: process_raw_data model_factory/pcaret_ts_model.py
	python model_factory/pcaret_ts_model.py --stock_ids TSLA \
	-clean_tech_data_store_dir data/clean_data/yfinancetechdata -start_date 2018-06-20 -end_date 2023-06-20 \
	-model_storage_path models/
 
#Build model simple time series
models/time_series_model.pkl: process_raw_data process_raw_gnews_data_extract_sentiment model_factory/auto_arima.py
	python model_factory/auto_arima.py --stock_ids TSLA \
	-clean_tech_data_store_dir data/clean_data/yfinancetechdata -start_date 2018-06-20 -end_date 2023-06-20 \
	-model_storage_path models/
#process data
#	Clean
#	transform
#	Feature generate if any
process_raw_data:  get_raw_tech_data data_processor/combine_tech_and_fundamentals.py data_processor/tech_indicator_processor.py
	python data_processor/tech_indicator_processor.py --stock_ids TSLA  -raw_tech_data_store_dir data/raw_data/yfinancetechdata \
	-clean_tech_data_store_dir data/clean_data/yfinancetechdata -start_date 2018-06-20 -end_date 2023-06-20

# Process and extract sentiment from the raw news data
process_raw_gnews_data_extract_sentiment:  get_raw_gnews_data sentiment_analyzer/analyze_n_append_gnews_sentiment.py
	python sentiment_analyzer/analyze_n_append_gnews_sentiment.py --stock_ids TSLA  -raw_tech_data_store_dir data/raw_data/googlenewsdata \
	-clean_tech_data_store_dir data/clean_data/googlenewsdata -start_date 2018-06-20 -end_date 2023-06-20

#collect gnews data
get_raw_eodnews_data:	scrapper/eodhistdata_scrapper.py
	python scrapper/eodhistdata_scrapper.py --stock_ids TSLA  -raw_tech_data_store_dir data/raw_data/eodnewsdata \
	-start_date 2018-06-20 -end_date 2023-06-20

#collect gnews data
get_raw_gnews_data:	scrapper/gnews_scrapper.py
	python scrapper/gnews_scrapper.py --stock_ids TSLA  -raw_tech_data_store_dir data/raw_data/googlenewsdata \
	-start_date 2018-06-20 -end_date 2023-06-20
#collect tech data
get_raw_tech_data:	scrapper/tech_indicator_scrapper.py
	python scrapper/tech_indicator_scrapper.py --stock_ids TSLA  -raw_tech_data_store_dir data/raw_data/yfinancetechdata \
	-start_date 2018-06-20 -end_date 2023-06-20