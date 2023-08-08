#Raw Data => Clean Data => Combined Data => Model
x_start_date = 2018-06-20
x_end_date = 2023-08-05
stocks = "TSLA"

all: models/time_series_model.pkl \
	models/pycaret_model.pkl models/pycaret_regression_model.pkl \
	models/ann.pkl models/lstm.pkl

models/lstm.pkl: process_raw_data model_factory/lstm.py
#models/lstm.pkl: model_factory/lstm.py 
	python model_factory/lstm.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

models/ann.pkl: process_raw_data model_factory/ann.py
	python model_factory/ann.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

models/pycaret_regression_model.pkl: process_raw_data model_factory/pycaret_regression_model.py
	python model_factory/pycaret_regression_model.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/


models/pycaret_model.pkl: process_raw_data model_factory/pcaret_ts_model.py
	python model_factory/pcaret_ts_model.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/
 
#Build model simple time series
models/time_series_model.pkl: process_raw_data process_raw_gnews_data_extract_sentiment model_factory/auto_arima.py
	python model_factory/auto_arima.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

# Correlations and granger causality as of now no dependency on process data but run it after that
corr_n_granger_causality: visual_causal_inference/granger_causality_check_for_features_n_label.py
	python visual_causal_inference/granger_causality_check_for_features_n_label.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined \
	-causality_storage_path data/clean_data/causal_inference/ \
	-start_date $(x_start_date) -end_date $(x_end_date)

#process data
#	Clean
#	transform
#	Feature generate if any

process_raw_data:  get_raw_tech_data process_raw_eodnews_data_extract_sentiment data_processor/combine_tech_and_fundamentals.py data_processor/tech_indicator_processor.py
	python data_processor/tech_indicator_processor.py --stock_ids $(stocks)  -raw_tech_data_store_dir data/raw_data/yfinancetechdata \
	-clean_tech_data_store_dir data/clean_data -start_date $(x_start_date) -end_date $(x_end_date)

# Process and extract sentiment from the raw news data
process_raw_gnews_data_extract_sentiment:  get_raw_gnews_data sentiment_analyzer/analyze_n_append_gnews_sentiment.py
	python sentiment_analyzer/analyze_n_append_gnews_sentiment.py --stock_ids $(stocks)  -raw_tech_data_store_dir data/raw_data/eodnewsdata \
	-clean_tech_data_store_dir data/clean_data/googlenewsdata -start_date $(x_start_date) -end_date $(x_end_date)

# Process and extract sentiment from the eod news data
process_raw_eodnews_data_extract_sentiment:  get_raw_eodnews_data sentiment_analyzer/analyze_eod_news_sentiment.py
	python sentiment_analyzer/analyze_eod_news_sentiment.py --stock_ids $(stocks)  -raw_tech_data_store_dir data/raw_data/eodnewsdata \
	-clean_tech_data_store_dir data/clean_data/eodnewsdata -start_date $(x_start_date) -end_date 2023-08-05

#collect eod data
get_raw_eodnews_data:	scrapper/eodhistdata_scrapper.py
	python scrapper/eodhistdata_scrapper.py --stock_ids $(stocks)  -raw_tech_data_store_dir data/raw_data/eodnewsdata \
	-start_date $(x_start_date) -end_date $(x_end_date)

#collect gnews data
get_raw_gnews_data:	scrapper/gnews_scrapper.py
	python scrapper/gnews_scrapper.py --stock_ids $(stocks)  -raw_tech_data_store_dir data/raw_data/googlenewsdata \
	-start_date $(x_start_date) -end_date $(x_end_date)
#collect tech data
get_raw_tech_data:	scrapper/tech_indicator_scrapper.py
	python scrapper/tech_indicator_scrapper.py --stock_ids $(stocks)  -raw_tech_data_store_dir data/raw_data/yfinancetechdata \
	-start_date $(x_start_date) -end_date $(x_end_date)