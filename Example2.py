"""
This example shows how to save a model and get predictive scores on a
separate data set
"""
from healthcareai import SupervisedModelTrainer
import pandas as pd
import time
import pickle


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

    # Divide into training set and scoring set

    df_train = df[df['InTestWindowFLG'] == 'N']
    df_to_score = df[df['InTestWindowFLG'] == 'Y']

    # Drop columns that won't help machine learning
    df_train = df_train.drop(['PatientID', 'InTestWindowFLG'], axis=1)

    # Establish training parameters
    train_params = {
        'predictiontype': 'classification',
        'predictedcol': 'ThirtyDayReadmitFLG',
        'graincol': 'PatientEncounterID',
    }

    # Train and Save random forest model
    save_filepath = 'example_rf.pkl'
    t2 = SupervisedModelTrainer(modeltype='rf', **train_params)
    t2.train(df_train, savepath=save_filepath)

    # Load random forest model from file
    rf_model = pickle.load(open(save_filepath, 'rb'))

    # Score scoring set & save results to SAM database
    df_scored = rf_model.score(df_to_score, saveto='sql')
    print('scored records:\n', df_scored.head())

    print('\nTime:\n', time.time() - t0)

if __name__ == "__main__":
    main()
