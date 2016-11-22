"""
Methods relating to database connections
"""
import pyodbc
from datetime import datetime


def write_scores_to_sql(scores_df,
                        predictiontype='classification',
                        graincol='PatientEncounterID',
                        server='localhost'):
    """
    Writes predictive scores to SQL Server based on
    :param scores_df:
    :param predictiontype:
    :param graincol:
    :param server:
    :return:
    """

    if predictiontype == 'classification':
        db_schema_table = '[SAM].[dbo].[HCPyDeployClassificationBASE]'
        predictedvalcol = 'PredictedProbNBR'
    else:
        db_schema_table = '[SAM].[dbo].[HCPyDeployRegressionBASE]'
        predictedvalcol = 'PredictedValueNBR'

    cecnxn = pyodbc.connect("""DRIVER={SQL Server Native Client 11.0};
                            SERVER=""" + server + """;
                            Trusted_Connection=yes;""")
    cursor = cecnxn.cursor()

    # The following allows output to work with datetime/datetime2
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    try:
        cursor.execute("""insert into """ + db_schema_table + """
                       (BindingID, BindingNM, LastLoadDTS, """ +
                       graincol + """,""" + predictedvalcol + """,
                       Factor1TXT, Factor2TXT, Factor3TXT)
                       values (0, 'PyTest', ?, ?, 0.98,
                       'FirstCol', 'SecondCol', 'ThirdCol')""",
                       (dt, int(scores_df[graincol].values[0])))
        cecnxn.rollback()

        print("\nSuccessfully inserted a test row into {}.".format(
            db_schema_table))
        print("SQL insert successfully rolled back (since it was a test).")

    except pyodbc.DatabaseError:
        print("\nFailed to insert values into {}.".format(
            db_schema_table))
        print("Check that the table exists with right col structure")
        print("Example column structure can be found in the docs")
        print("Your GrainID col might not match that in your input table")

    finally:
        try:
            cecnxn.close()
        except pyodbc.DatabaseError:
            print("""\nAn attempt to complete a transaction has failed.
                    No corresponding transaction found. \nPerhaps you don''t have
                    permission to write to this server.""")

    # Convert to base int and float instead of numpy data type for SQL insert
    scores_df[graincol] = scores_df[graincol].astype(int)
    scores_df['y_pred'] = scores_df['y_pred'].astype(float)

    # Add hc specific columns
    scores_df['BindingID'] = 0
    scores_df['BindingNM'] = 'Python'
    scores_df['LastLoadDTS'] = datetime.utcnow().strftime(
        '%Y-%m-%d %H:%M:%S.%f')[:-3]
    scores_df[predictedvalcol] = scores_df['y_pred']

    write_col_names = ['BindingID', 'BindingNM', 'LastLoadDTS',
                       'PatientEncounterID', predictedvalcol,
                       'Factor1TXT', 'Factor2TXT', 'Factor3TXT']

    # Convert to tuple matrix format
    to_write = tuple((tuple(row) for row in scores_df[write_col_names].as_matrix()))

    # if debug:
    #     print('\nTop rows of 2-d list immediately before insert into db')
    #     print(pd.DataFrame(output_2dlist[0:3]).head())

    cecnxn = pyodbc.connect("""DRIVER={SQL Server Native Client 11.0};
                                       SERVER=""" + server + """;
                                       Trusted_Connection=yes;""")
    cursor = cecnxn.cursor()

    try:
        cursor.executemany("""insert into """ + db_schema_table + """
                                   (BindingID, BindingNM, LastLoadDTS, """ +
                           graincol + """,""" + predictedvalcol + """,
                                   Factor1TXT, Factor2TXT, Factor3TXT)
                                   values (?,?,?,?,?,?,?,?)""", to_write)
        cecnxn.commit()

        # Todo: count and display (via pyodbc) how many rows inserted
        print("\nSuccessfully inserted rows into {}.".
              format(db_schema_table))

    except pyodbc.DatabaseError:
        print("\nFailed to insert values into {}.".
              format(db_schema_table))
        print("Was your test insert successful earlier?")
        print("If so, what has changed with your entity since then?")

    finally:
        try:
            cecnxn.close()
        except pyodbc.DatabaseError:
            print("""\nAn attempt to complete a transaction has failed.
                          No corresponding transaction found. \nPerhaps you don't
                          have permission to write to this server.""")

