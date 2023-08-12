#Raw Data => Clean Data => Combined Data => Model
### Please run run_TSLA_F_GM_TM_all_combination_train_evaluate unless you have eodhist news API Key###
### To prepare everythign from scratch get the api add the same under scrapped/eod_xxxx scrapper######
### Replace dependencies of models on the process_raw_data for all models then ***run make all***#####

x_start_date = 2018-06-20
x_end_date = 2023-08-05
stocks += $(STOCK)

all: models/time_series_model.pkl \
	models/pycaret_ts_model.pkl models/pycaret_regression_model.pkl \
	models/ann.pkl models/lstm.pkl

models/evaluate_pycaret_regression_model: model_factory/evaluate_pycaret_regression_model.py
	python model_factory/evaluate_pycaret_regression_model.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

models/evaluate_vanilla_lstm: model_factory/evaluate_lstm_vanilla_univariate_log_return_n_step.py
	python model_factory/evaluate_lstm_vanilla_univariate_log_return_n_step.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

models/vanilla_lstm: model_factory/lstm_vanilla_univariate_log_return_n_step.py
	python model_factory/lstm_vanilla_univariate_log_return_n_step.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

models/evaluate_ann: model_factory/evaluate_ann.py
	python model_factory/evaluate_ann.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

models/evaluate_lstm: model_factory/evaluate_lstm.py
#models/lstm.pkl: model_factory/lstm.py 
	python model_factory/evaluate_lstm.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/


#models/lstm.pkl: process_raw_data model_factory/lstm.py
models/lstm.pkl: model_factory/lstm.py

#models/lstm.pkl: model_factory/lstm.py 
	python model_factory/lstm.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

#models/ann.pkl: process_raw_data model_factory/ann.py
models/ann.pkl: model_factory/ann.py
	python model_factory/ann.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

#models/pycaret_regression_model.pkl: process_raw_data model_factory/pycaret_regression_model.py
models/pycaret_regression_model.pkl: model_factory/pycaret_regression_model.py
	python model_factory/pycaret_regression_model.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/

#models/pycaret_ts_model.pkl: process_raw_data model_factory/pcaret_ts_model.py
models/pycaret_ts_model.pkl: model_factory/pcaret_ts_model.py
	python model_factory/pcaret_ts_model.py --stock_ids $(stocks) \
	-clean_tech_data_store_dir data/clean_data/all_combined -start_date $(x_start_date) -end_date $(x_end_date) \
	-model_storage_path models/
 
#Build model simple time series
#models/time_series_model.pkl: process_raw_data process_raw_gnews_data_extract_sentiment model_factory/auto_arima.py
models/time_series_model.pkl: process_raw_data model_factory/auto_arima.py

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
#process_raw_gnews_data_extract_sentiment:  get_raw_gnews_data sentiment_analyzer/analyze_n_append_gnews_sentiment.py
#	python sentiment_analyzer/analyze_n_append_gnews_sentiment.py --stock_ids $(stocks)  -raw_tech_data_store_dir data/raw_data/eodnewsdata \
#	-clean_tech_data_store_dir data/clean_data/googlenewsdata -start_date $(x_start_date) -end_date $(x_end_date)

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