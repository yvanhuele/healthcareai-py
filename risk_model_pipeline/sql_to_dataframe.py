from dotenv import find_dotenv, load_dotenv
import pyodbc
import pandas as pd
import os
from time import time

def sql_to_dataframe(sql):
	''' Use the .env file in the risk_models directory to get the necessary variables
	to connect to CAFE EDW and CAFE SSISDB
	'''
	os.chdir("C:\\Users\\aaronn\\repos\\cafe\\risk_models")

	load_dotenv(find_dotenv())  

	server = os.environ.get("CAFE_Azure_SQLDW_Server")
	database = os.environ.get("CAFE_Azure_SQLDW_Database")
	username = os.environ.get("CAFE_Azure_SQLDW_Username")
	password = os.environ.get("CAFE_Azure_SQLDW_Password")
	driver = '{ODBC Driver 13 for SQL Server}'
	cnxn = pyodbc.connect('DRIVER=' + driver + ';PORT=1433;SERVER=' + server + ';PORT=1443;DATABASE=' + database + ';UID=' + username + ';PWD=' + password)

	start = time()
	df = pd.read_sql(sql, cnxn)
	load_time = time() - start
	data_output = {'data':df, 'load_time':load_time, 'sql':sql }
	cnxn.close()
	return data_output



	