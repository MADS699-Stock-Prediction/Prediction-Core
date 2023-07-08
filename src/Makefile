#Raw Data => Clean Data => Combined Data => Model

all: models/time_series_model.pkl

#Build model simple time series
models/time_series_model.pkl: process_raw_data model_factory/auto_arima.py
	python model_factory/auto_arima.py --stock_ids TSLA \
	-clean_tech_data_store_dir data/clean_data/yfinancetechdata -start_date 2018-06-20 -end_date 2023-06-20 \
	-model_storage_path models/
#process data
#	Clean
#	transform
#	Feature generate if any
process_raw_data:  get_raw_data data_processor/tech_indicator_processor.py
	python data_processor/tech_indicator_processor.py --stock_ids TSLA  -raw_tech_data_store_dir data/raw_data/yfinancetechdata \
	-clean_tech_data_store_dir data/clean_data/yfinancetechdata -start_date 2018-06-20 -end_date 2023-06-20

#collect data
get_raw_data:	scrapper/tech_indicator_scrapper.py
	python scrapper/tech_indicator_scrapper.py --stock_ids TSLA  -raw_tech_data_store_dir data/raw_data/yfinancetechdata \
	-start_date 2018-06-20 -end_date 2023-06-20