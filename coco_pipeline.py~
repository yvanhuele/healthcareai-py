''' Risk Model Pipeline

	This pipeline is designed to be run separately for each dataset/metric set (i.e. HCUP NIS).
	
	MVP:
		[x] Load training and customer data from SQL statements
		[ ] Run compare_datasets (output to logging, stop process if there are issues) - AN working on this
		[ ] Data Pre-processing (training and customer datasets)
		[ ] Model Selection and Evaluation (log evaluation metrics for each model type).  Select best model
			for re-training.
		[ ] Re-train overall best model on Full training dataset (save model to pickle file)
		[ ] Apply final model to customer dataset (read saved model from pickle, save expected values to new 
			dataframe with FacilityAccountID, and log evaluation metrics)
		[ ] Write model output dataframe back to CAFE EDW
'''

from risk_model_pipeline import sql_to_dataframe, dataframe_to_sql
from risk_model_pipeline.compare_datasets import data_type_mismatch
import numpy as np
import pandas as pd
import healthcareai as hc
from healthcareai.common.transformers import DataFrameImputer, ImpactCoding, FeatureScaling
import json

## LOAD CONFIG
file = r'C:\Users\coco\healthcareai-py\risk_model_pipeline\config_pipeline.json'
with open(file) as json_data:
    d = json.load(json_data)
    print(d)
print(d)
print(d['Mortality'])
## LOAD DATASETS
def generate_load_queries(d, n_rows=None):
	columns = d['FeatureColumns'] + d['TargetColumn']
	grain_col = d['CustomerGrainColumn']
	train_table = d['TrainTable']
	customer_table = d['CustomerTable']
	column_str = ', '.join(str(i) for i in columns)

	if n_rows != None:
		train_query = "SELECT TOP " + str(n_rows) + " " + column_str + " FROM " + train_table
		customer_query = "SELECT TOP " + str(n_rows) + " " + grain_col + ", " + column_str + " FROM " + customer_table
	else:
		train_query = "SELECT " + column_str + " FROM " + train_table
		customer_query = "SELECT " + grain_col + ", " + column_str + " FROM " + customer_table

	return train_query, customer_query

train_query, customer_query = generate_load_queries(d['Mortality'], n_rows=1000)
print(train_query)
print(customer_query)
train = sql_to_dataframe.sql_to_dataframe(train_query)
customer = sql_to_dataframe.sql_to_dataframe(customer_query)
print('Training data load time: %.2f' % train['load_time'])
print('Customer data load time: %.2f' % customer['load_time'])
print('\n')
train_df = train['data']
customer_df = customer['data']
print("Training dataset shape: ", train_df.shape)
print("Customer dataset shape: ", customer_df.shape)
	
mismatch_datatypes = data_type_mismatch(train_df, customer_df)
print('Data Types')
print(train_df.dtypes)
print('\n')
print(customer_df.dtypes)
print('\n')
print("Columns with mismatched data types: ")
print(mismatch_datatypes)
print('\n')

# missing_values = valid_value_compare(train_df, customer_df)
# print ('Valid Value Comparison (values in training but not customer)')
# print (missing_values)
# print('\n')


## DATA PROCESSING
print("IMPACT CODING")
impact = ImpactCoding(columns='MSDRGNormCD', target='DiedFLG')
impact.fit(train_df)
impact.transform(train_df)
# impact.transform(customer_df)
print("Training dataset shape: ", train_df.shape)
print("Customer dataset shape: ", customer_df.shape)

# scale = FeatureScaling(columns=["MSDRGNormCD","LengthOfStayDaysNBR"])
# scale.fit(train_df)
# scale.transform(train_df)
# scale.transform(customer_df)




oo = hc.DevelopSupervisedModel(dataframe = train_df, predicted_column = 'DiedFLG', model_type = 'classification')
# Imputation

# # # Impact Coding
# # print(train_df.DiedFLG.value_counts(dropna=False))
# # print(train_df.dtypes)
# # train_df['DiedFLG'] = train_df['DiedFLG'].astype(int)
# # print(train_df.dtypes)
# # oo.dataframe = impact_coding_on_a_single_column(dataframe = oo.dataframe, predicted_column = 'DiedFLG', impact_column = 'MSDRGNormCD')


# impact = ImpactCoding()
# print(train_df.shape)
# train_df = impact.fit(train_df,'DiedFLG','MSDRGNormCD')
# train_df = impact.transform(train_df,'MSDRGNormCD')
# print(train_df.shape)
# print(train_df.MSDRGNormCD.head())
# #customer_df = impact.transform(customer_df,'MSDRGNormCD')

# # Under-sampling
# print("Under-sampling")
# print(oo.dataframe.shape)
# oo.under_sampling()
# print(oo.dataframe.shape)

# # # Feature Scaling
# # oo.train_test_split()
# # oo.feature_scaling(['LengthOfStayDaysNBR','DRG_impact_coded'])
# # print("features scaled")


# # ## MODEL SELECTION

# # ## APPLY MODEL TO CUSTOMER DATA

# # ## WRITE TO CAFE
# # # # output = customer_df[['DiedFLG','GenderNormDSC']].copy().as_matrix().tolist()
# # # # dataframe_to_sql('RiskOutput.test_prediction', output)
# # # # print("Completed load")


# # # oo.ensemble_classification()
# # # oo.write_classification_metrics_to_json()
# # # oo.write_model_to_pickle('classification_best_model')
# # # #model_reread = pickle.load(open("classification_best_model.pkl", "rb" ) )

