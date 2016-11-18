import pandas as pd
from sklearn.base import TransformerMixin


class DateTimeColumnFilter(TransformerMixin):
    def __init__(self):
        """
        Removes columns with suffix 'DTS'
        """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # first step in the pipeline copies the df
        newX = X.copy()
        cols = [c for c in X.columns if c[-3:] != 'DTS']
        return newX[cols]


class GrainColumnFilter(TransformerMixin):
    def __init__(self, graincolname):
        """
        Removes the grain column from the data
        :param graincolname: the grain column (e.g. PatientID)
        """
        self.graincolname = graincolname

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols = [c for c in X.columns if c != self.graincolname]
        return X[cols]


class NullValueFilter(TransformerMixin):
    def __init__(self, ignorecols=[]):
        """
        Removes any rows with Null values
        :param ignorecols: an array of column names to ignore
        """
        self.ignorecols = ignorecols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        subset = [c for c in X.columns if c not in self.ignorecols]
        X = X.dropna(subset=subset)
        return X
