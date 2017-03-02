import sys
import datetime
import math
import pyodbc
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from healthcareai.common.transformers import DataFrameImputer
from healthcareai.common import model_eval
from healthcareai.common import filters
import numpy as np
import pandas as pd


class DeploySupervisedModel(object):
    def __init__(self, modeltype, dataframe, graincol, windowcol, predictedcol, impute, debug=False):
        self.modeltype = modeltype
        self.dataframe = dataframe
        self.graincol = graincol
        self.impute = impute
        self.y_pred = None  # Only used for unit test

        if debug:
            print('\nTypes of original dataframe:')
            print(dataframe.dtypes)
            print('\nShape and top 5 rows of original dataframe:')
            print(dataframe.shape)
            print(dataframe.head())

        self.dataframe = filters.remove_datetime_columns(self.dataframe)

        if debug:
            print('\nDataframe after removing DTS columns:')
            print(dataframe.head())
            print('\nNow either doing imputation or dropping rows with NULLs')
            print('\nTotal df shape before impute/remove')
            print(dataframe.shape)

        pd.options.mode.chained_assignment = None  # default='warn'

        # TODO: put try/catch here when type = class and pred col is numer
        dataframe.replace('NULL', np.nan, inplace=True)

        if debug:
            print('\nTotal df after replacing NULL with nan')
            print(dataframe.shape)
            print(dataframe.head())

        if self.impute:
            dataframe = DataFrameImputer().fit_transform(dataframe)

            if debug:
                print('\nTotal df after doing imputation:')
                print(dataframe.shape)
                print(dataframe.head())
        else:
            # TODO switch similar statements to work inplace
            # If not impute, only remove rows in train that contain NULLS
            if debug:
                print('\nTrain portion before impute/remove')
                print(dataframe.ix[dataframe[windowcol] == 'N'].shape)

            df_train_temp = dataframe.ix[dataframe[windowcol] == 'N']
            df_test_temp = dataframe.ix[dataframe[windowcol] == 'Y']
            df_train_temp.dropna(axis=0, how='any', inplace=True)

            # Recombine same (unaltered) test rows with modified training set
            dataframe = pd.concat([df_test_temp, df_train_temp])

            if debug:
                print('\nTotal df after dropping rows with NULLS:')
                print(dataframe.shape)
                # noinspection PyUnresolvedReferences
                print(dataframe.head())
                print('\nTrain portion after removing training rows w/ NULLs')
                # noinspection PyUnresolvedReferences,PyUnresolvedReferences
                print(dataframe.ix[dataframe[windowcol] == 'N'].shape)

        # Call new function!!
        # Convert predicted col to 0/1 (otherwise won't work with GridSearchCV)
        # Makes it so that healthcareai only handles N/Y in pred column
        if modeltype == 'classification':
            # Turning off warning around replace
            # pd.options.mode.chained_assignment = None  # default='warn'
            # TODO: put try/catch here when type = class and pred col is numer
            dataframe[predictedcol].replace(['Y', 'N'], [1, 0], inplace=True)

            if debug:
                print("""\nDataframe after converting to 1/0 instead of Y/N
                for classification:""")
                print(dataframe.head())

        # Split graincol into test piece; is taken out of ML and used in dfout
        self.graincol_test = dataframe[graincol].loc[dataframe[windowcol] == 'Y']
        pd.options.mode.chained_assignment = None  # default='warn'
        # TODO: Fix this copy of a slice SettingWithCopyWarning:

        dataframe.drop(graincol, axis=1, inplace=True)

        if debug:
            print('\nGrain_test col after splitting it from main df')
            print(np.shape(self.graincol_test))
            print(self.graincol_test.head())

            print('\ndf after splitting out graincol and before creating dums')
            print(dataframe.head())

        # Create dummy vars for all cols but predictedcol
        # First switch (temporarily) pred col to numeric (so it's not dummy)
        dataframe[predictedcol] = pd.to_numeric(arg=dataframe[predictedcol], errors='raise')
        dataframe = pd.get_dummies(dataframe, drop_first=True, prefix_sep='.')

        if debug:
            print('\nDataframe after creating dummy vars:')
            print(dataframe.head())

        # Split train off of maindf
        # noinspection PyUnresolvedReferences
        df_train = dataframe.ix[dataframe[windowcol + '.Y'] == 0]

        if debug:
            print('\nTraining dataframe right after splitting it off:')
            print(dataframe.head())

        # Always remove rows where predicted col is NULL in train
        df_train.dropna(subset=[predictedcol], how='all', inplace=True)
        if debug:
            print('\nTraining df after removing rows where pred col is NULL:')
            print(df_train.head())
            print('\nSplitting off test set from main df...')

        # Split test off of main df
        # noinspection PyUnresolvedReferences
        df_test = dataframe.ix[dataframe[windowcol + '.Y'] == 1]

        if debug:
            print('\nTest set after splitting off of main df:')
            print(df_test.head())

        # Drop window col from train and test
        df_train = df_train.drop(windowcol + '.Y', axis=1)

        if debug:
            print('\nTrain set after dropping window col (from training set):')
            print(df_train.head())

        df_test = df_test.drop(windowcol + '.Y', axis=1)

        if debug:
            print('\nTest set after dropping window col (from test set):')
            print(df_test.head())

        # Check user input and remove rows with NULLS if not doing imputation
        self.impute = impute

        self.X_train = df_train.drop([predictedcol], axis=1)
        self.y_train = np.squeeze(df_train[[predictedcol]])

        self.X_test = df_test.drop([predictedcol], axis=1)
        self.y_test = np.squeeze(df_test[[predictedcol]])

        if debug:
            print('\nShape of X_train, y_train, X_test, and y_test:')
            print(self.X_train.shape)
            print(self.y_train.shape)
            print(self.X_test.shape)
            print(self.y_test.shape)

    def deploy(self, method, cores, server, dest_db_schema_table, trees=200, mtry=None, use_saved_model=False,
               debug=False):

        if debug:
            print("\ngraincol test shape and cell type before db prelim-insert check")
            print(np.shape(self.graincol_test))
            print(type(self.graincol_test.iloc[0]))

        cecnxn = pyodbc.connect("""DRIVER={SQL Server Native Client 11.0};
                                   SERVER=""" + server + """;
                                   Trusted_Connection=yes;""")
        cursor = cecnxn.cursor()

        if self.modeltype == 'classification':
            predictedvalcol = 'PredictedProbNBR'
        else:
            predictedvalcol = 'PredictedValueNBR'

        # First, check the connection by inserting test data (and rolling back)
        self._test_database_connection(cecnxn, dest_db_schema_table, predictedvalcol)

        if self.modeltype == 'classification' and method == 'linear':
            algorithm = LogisticRegression(n_jobs=cores)

            self.y_pred = model_eval.clfreport(
                modeltype=self.modeltype,
                debug=debug,
                devcheck='notdev',
                algo=algorithm,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                use_saved_model=use_saved_model)

        elif self.modeltype == 'regression' and method == 'linear':
            algorithm = LinearRegression(n_jobs=cores)

            self.y_pred = model_eval.clfreport(
                modeltype=self.modeltype,
                debug=debug,
                devcheck='notdev',
                algo=algorithm,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                use_saved_model=use_saved_model)

        if self.modeltype == 'classification' and method == 'rf':

            # TODO: think about moving this to model_eval mtry function
            if not mtry:
                mtry = math.floor(math.sqrt(len(self.X_train.columns.values)))

            algorithm = RandomForestClassifier(n_estimators=trees, max_features=mtry, n_jobs=cores,
                                               verbose=(2 if debug is True else 0))

            self.y_pred = model_eval.clfreport(
                modeltype=self.modeltype,
                debug=debug,
                devcheck='notdev',
                algo=algorithm,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                use_saved_model=use_saved_model)

        elif self.modeltype == 'regression' and method == 'rf':

            # TODO: think about moving this to model_eval mtry function
            if not mtry:
                mtry = math.floor(len(self.X_train.columns.values) / 3)

            algorithm = RandomForestRegressor(n_estimators=trees, max_features=mtry, n_jobs=cores,
                                              verbose=(2 if debug is True else 0))

            self.y_pred = model_eval.clfreport(modeltype=self.modeltype, debug=debug, devcheck='notdev', algo=algorithm,
                                               X_train=self.X_train, y_train=self.y_train, X_test=self.X_test,
                                               use_saved_model=use_saved_model)

        # Calculate three imp columns
        first_fact, second_fact, third_fact = model_eval. \
            findtopthreefactors(debug,
                                self.X_train,
                                self.y_train,
                                self.X_test,
                                self.modeltype,
                                use_saved_model)

        # Convert to base int instead of numpy data type for SQL insert
        graincol_baseint = [int(self.graincol_test.iloc[i])
                            for i in range(0, len(self.graincol_test))]
        y_pred_baseint = [float(self.y_pred[i])
                          for i in range(0, len(self.y_pred))]

        # Create columns for export to SQL Server
        X_test_length = len(self.X_test.iloc[:, 0])
        bindingid = [0] * X_test_length
        bindingnm = ['Python'] * X_test_length

        # Create vector with time to the millisecond
        lastloaddts = [datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]] * X_test_length

        # Put everything into 2-d list for export
        output_2dlist = list(zip(bindingid,
                                 bindingnm,
                                 lastloaddts,
                                 graincol_baseint,
                                 y_pred_baseint,
                                 first_fact,
                                 second_fact,
                                 third_fact))

        if debug:
            print('\nTop rows of 2-d list immediately before insert into db')
            print(pd.DataFrame(output_2dlist[0:3]).head())

        try:
            cursor.executemany("""insert into """ + dest_db_schema_table + """
                               (BindingID, BindingNM, LastLoadDTS, """ +
                               self.graincol + """,""" + predictedvalcol + """,
                               Factor1TXT, Factor2TXT, Factor3TXT)
                               values (?,?,?,?,?,?,?,?)""", output_2dlist)
            cecnxn.commit()

            # Todo: count and display (via pyodbc) how many rows inserted
            print("\nSuccessfully inserted rows into {}.".
                  format(dest_db_schema_table))

        except pyodbc.DatabaseError:
            print("\nFailed to insert values into {}.".
                  format(dest_db_schema_table))
            print("Was your test insert successful earlier?")
            print("If so, what has changed with your entity since then?")

        finally:
            try:
                cecnxn.close()
            except pyodbc.DatabaseError:
                print("""\nAn attempt to complete a transaction has failed.
                      No corresponding transaction found. \nPerhaps you don't
                      have write permission on this server.""")
            finally:
                sys.exit()

    def _test_database_connection(self, cecnxn, dest_db_schema_table, predictedvalcol):
        """ Test if the database connection is valid by inserting a row """
        # The following allows output to work with datetime/datetime2
        date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        try:
            cursor = cecnxn.cursor()
            insertion_query = "insert into {} (BindingID, BindingNM, LastLoadDTS, {}, {}, Factor1TXT, Factor2TXT, " \
                              "Factor3TXT) values (0, 'PyTest', ?, ?, 0.98, 'FirstCol', 'SecondCol', " \
                              "'ThirdCol')".format(dest_db_schema_table, self.graincol, predictedvalcol)
            cursor.execute(insertion_query, (date, int(self.graincol_test.iloc[0])))
            cecnxn.rollback()

            print("\nSuccessfully inserted a test row into {}.".format(dest_db_schema_table))
            print("SQL insert successfully rolled back (since it was a test).")

        except pyodbc.DatabaseError:
            print("\nFailed to insert values into {}.".format(dest_db_schema_table))
            print("Check that the table exists with right column structure")
            print("Example column structures can be found in the docs")
            print("Your GrainID col might not match that in your input table")

        finally:
            try:
                cecnxn.close()
            except pyodbc.DatabaseError:
                print("""\nAn attempt to complete a transaction has failed.
                No corresponding transaction found. \nPerhaps you don't have write
                permission on this server.""")
            finally:
                sys.exit()
