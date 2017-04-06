from dotenv import find_dotenv, load_dotenv
import pyodbc
import pandas as pd
import os
from time import time

def dataframe_to_sql(db_schema_table, dataframe):

	os.chdir("C:\\Users\\aaronn\\repos\\cafe\\risk_models")

	load_dotenv(find_dotenv())  

	server = os.environ.get("CAFE_Azure_SQLDW_Server")
	database = os.environ.get("CAFE_Azure_SQLDW_Database")
	username = os.environ.get("CAFE_Azure_SQLDW_Username")
	password = os.environ.get("CAFE_Azure_SQLDW_Password")
	driver = '{ODBC Driver 13 for SQL Server}'
	cnxn = pyodbc.connect('DRIVER=' + driver + ';PORT=1433;SERVER=' + server + ';PORT=1443;DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
	cursor = cnxn.cursor()
	print("About to insert data")
	cursor.executemany("""insert into """ + db_schema_table + """(DiedFLG, GenderNormDSC) values (?,?)""", dataframe)
	cnxn.commit()
	cnxn.close()
	return