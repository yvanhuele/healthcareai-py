"""This file demonstrates how to train and compare two models on the same data
set, as well as examples of reading form csv and SQL Server. Note that this
example can be run as-is after installing healthcareai.  Example 2 will
demonstrate how to get scores from a saved model.
"""
from healthcareai import SupervisedModelTrainer
import pandas as pd
import time

def main():

    t0 = time.time()

    # CSV snippet for reading data into dataframe
    df = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv',
                     na_values=['None'])

    # SQL snippet for reading data into dataframe
    # import pyodbc
    # cnxn = pyodbc.connect("""SERVER=localhost;
    #                          DRIVER={SQL Server Native Client 11.0};
    #                          Trusted_Connection=yes;
    #                          autocommit=True""")
    #
    # df = pd.read_sql(
    #     sql="""SELECT
    #            *
    #            FROM [SAM].[dbo].[HCPyDiabetesClinical]""",
    #     con=cnxn)
    #
    # # Set None string to be None type
    # df.replace(['None'],[None],inplace=True)

    # Look at data that's been pulled in
    print(df.head())
    print(df.dtypes)

    # Drop columns that won't help machine learning
    df.drop(['PatientID','InTestWindowFLG'], axis=1, inplace=True)

    # Establish training parameters
    train_params = {
        'predictiontype': 'classification',
        'predictedcol': 'ThirtyDayReadmitFLG',
        'graincol': 'PatientID',
    }

    # Train Linear Classifier
    t1 = SupervisedModelTrainer(modeltype='linear', **train_params)
    linear_model = t1.train(df)

    # Train Random Forest Classifier
    t2 = SupervisedModelTrainer(modeltype='rf', **train_params)
    rf = t2.train(df)

    #compare
    print('\nauc of linear model: ', linear_model.get_roc_auc())
    print('auc of rf model: ', rf.get_roc_auc())

    # Look at rf feature importance rankings
    # print(rf.get_top_features())

    print('\nTime:\n', time.time() - t0)

if __name__ == "__main__":
    main()