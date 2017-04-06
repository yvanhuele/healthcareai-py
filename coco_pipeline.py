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
import json

from sklearn.model_selection import train_test_split

from functions import DataFrameImputer
from functions import ImpactCoding
from functions import impact_coding
from functions import impact_coding_learn
from functions import impact_coding_transform
from functions import under_sampling
from functions import over_sampling
from functions import train_test_splt
from functions import feature_scaling
from functions import FeatureScaling
from functions import randomsearch
from functions import pick_best_model

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

print('\n')
print("Training dataset shape: ", train_df.shape)
print("Customer dataset shape: ", customer_df.shape)
	
mismatch_datatypes = data_type_mismatch(train_df, customer_df)

print('\n')
print('train_df Data Types:')
print(train_df.dtypes)
print('\n')
print('customer_df Data Types:')
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

# (1) Imputation -- No need for imputation here

# (2) Impact Coding:

# train, test = train_test_split(train_df,
#                                test_size=0.8,
# 			       random_state = 0)
# train = train.copy()
# test = test.copy()

impact = ImpactCoding(columns='MSDRGNormCD', target='DiedFLG')
impact.fit(train_df)
impact.transform(train_df)

# (3) Dummies -- No need for dummies here

# (4) Under/Over sampling:
train_df = under_sampling(train_df,'DiedFLG')

# (5) Train/Test Split:
X_train, X_test, y_train, y_test = \
    train_test_splt(train_df,predicted_column = 'DiedFLG')

# (6) Feature Scaling:
scaler = FeatureScaling(columns = ["MSDRGNormCD",
                                   "LengthOfStayDaysNBR"])
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

# MODEL SELECTION:

estimator_dictionary = randomsearch(X_train,y_train,'classification')

best_model = pick_best_model(X_test,y_test,estimator_dictionary)

# Retrain best model on all data:

Xframes = [X_train, X_test]
yframes = [y_train,y_test]

X = pd.concat(Xframes)
y = pd.concat(yframes)
best_model.fit(X,y)


# # ## APPLY MODEL TO CUSTOMER DATA

# # ## WRITE TO CAFE
# # # # output = customer_df[['DiedFLG','GenderNormDSC']].copy().as_matrix().tolist()
# # # # dataframe_to_sql('RiskOutput.test_prediction', output)
# # # # print("Completed load")


# # # oo.ensemble_classification()
# # # oo.write_classification_metrics_to_json()
# # # oo.write_model_to_pickle('classification_best_model')
# # # #model_reread = pickle.load(open("classification_best_model.pkl", "rb" ) )

