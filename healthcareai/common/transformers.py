from sklearn.base import TransformerMixin
from healthcareai.common.types import is_categorical
import pandas as pd
import numpy as np


class DataFrameImputer(TransformerMixin):

    def __init__(self, impute=True):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
        self.impute = impute
        self.obj_list = None
        self.fill = None

    def fit(self, X, y=None):
        if self.impute:
            # Grab list of object column names before doing imputation
            self.obj_list = X.select_dtypes(include=['object']).columns.values

            self.fill = pd.Series([X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                index=X.columns)
        else:
            pass
        return self

    def transform(self, X, y=None):
        if self.impute:
            print('Imputing missing data....')
            X = X.fillna(self.fill)

            for i in self.obj_list:
                X[i] = X[i].astype(object)
        else:
            pass
        return X


class Dummifier(TransformerMixin):

    def __init__(self, ignorecols=[], categorical=[]):
        """Creates dummy columns from categorical variables

        :params ignorecols: a list of column names to ignore
        :params categorical: a list of column names to treat as categorical
            (for all other columns categorical vs continuous will be inferred)

        """
        self.ignorecols = ignorecols
        self.categorical = categorical

    def fit(self, X, y=None):
        """
        Stores a list of columns and values to dummify
        """
        self.dummy_dict = {}
        categorical_cols = [c for c in X.columns if
                            (is_categorical(X[c]) or \
                             c in self.categorical) and (
                            c not in self.ignorecols)]
        for col in categorical_cols:
            self.dummy_dict[col] = X[col].unique()
        return self

    def transform(self, X, y=None):
        for colname, vals in self.dummy_dict.items():
            for val in vals:
                X['%s__%s' % (colname, val)] = (X[colname] == val).astype(
                    int)
        X = X.drop(self.dummy_dict.keys(), axis=1)
        return X
