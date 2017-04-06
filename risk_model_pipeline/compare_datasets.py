'''
	Compare_Datasets:
		MVP:	
			[x] Read in two datasets (training and customer datasets)
			[ ] Extra Columns (check both datasets and print any extra column names as tableNM.columnNM
			[ ] For each matching column compare data types. If they don't match write ColumnNM, train_datatype, cusotmer_datatype
			[ ] For each non-numeric matching column compare valid values between train and customer datasets.
				If a value is missing from a given column, print (Dataset, ColumnNM, Missing Value)
		Advanced:
			[ ] Compare relative frequency of categories for categorical variables
			[ ] Compare distributions for continuous variables
			[ ] Compare relative frequency of missing values
'''
			
			
def compare_lists(list_1, list_2):
	''' Compare two lists to see if they contain the same items.  If list_1
		contains items not in list_2, return those items as a list.	
	'''
	extra_items = [k for k in list_1 if k not in list_2]
	return extra_items

def data_type_mismatch(df_a, df_b):
	'''	Inspects all columns that share the same column name
		between two dataframes and returns the column name if 
		the data types between the two dataframes for that column
		don't match
	'''
	mismatch_dtypes = []
	for col in df_a.columns.tolist():
		if col in df_b.columns.tolist():
			if df_a[col].dtype == df_b[col].dtype:
				continue
			else:
				mismatch_dtypes.append(col)
	
	return mismatch_dtypes
	
def valid_value_compare(df_a, df_b):
	''' Compares the valid values of all columns that share the same column
		name between to dataframes.  Returns column names and unique values
		missing from each dataset.	Only looks at columns with matching names
		and data types.
	'''
	df_a_columns = df_a.columns.tolist()
	df_b_columns = df_b.columns.tolist()
	unique_columns_a = compare_lists(df_a_columns, df_b_columns)
	common_cols = list(set(df_a_columns) - set(unique_columns_a))
	
	missing_values = {}
	df_a
	for col in common_cols:
		if df_a[col].dtype == df_b[col].dtype:
			data_type = df_a[col].dtype
			if data_type == '|O':
				unique_values_a = df_a[col].unique()
				unique_values_b = df_b[col].unique()
				
				extra_values = list(set(unique_values_a) - set(unique_values_b))
				missing_values[col] = extra_values
			else:
				continue
			
	return missing_values