from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


class DataFrameImputer(TransformerMixin):
    """Impute missing values.

    Columns of dtype object are imputed with the most frequent value in column.

    Columns of other types are imputed with mean of column.
    """
    def __init__(self):
        self.obj_list = None
        self.fill = None

    def fit(self, X, y=None):
        # Grab list of object column names before doing imputation
        self.obj_list = X.select_dtypes(include=['object']).columns.values

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        X.fillna(self.fill,inplace=True)

        for i in self.obj_list:
            X[i] = X[i].astype(object)

        # return self for scikit compatibility
        return X

class ImpactCoding(object):
    def __init__(self,columns,target):
        if type(columns) == str:
            columns = [columns]
        self.columns = columns
        self.target = target
        self.x_bar = None
        self.dics = []
        
    def fit(self,df):
        self.x_bar = df[self.target].mean()
        for col0 in self.columns:
            dic = {}
            for category in df[col0].unique():
                sub_dataframe = df[df[col0] == category]
                dic[category] = sub_dataframe[self.target].mean()
            self.dics.append(dic)
        return self        

    def transform(self,df):
        for ii in range(0,len(self.columns)):
            col0 = self.columns[ii]
            dic0 = self.dics[ii]
            df[col0] = np.where(
                                df[col0].isin(dic0.keys()),
                                df[col0].replace(dic0),
	                        self.x_bar) \
                                - self.x_bar
        return df

    def fit_transform(self,df):
        self.fit(df)
        self.transform(df)

class FeatureScaling(object):
    def __init__(self,columns):
        if type(columns) == str:
            columns = [columns]
        self.columns = columns
        self.means = None
        self.stds = None

    def fit(self,df):
        self.means = []
        self.stds = []
        for ii in range(0,len(self.columns)):
            col0 = self.columns[ii]
            self.means.append(df[col0].mean())
            self.stds.append(df[col0].std())
        return self

    def transform(self,df):
        for ii in range(0,len(self.columns)):
            col0 = self.columns[ii]
            mean0 = self.means[ii]
            std0 = self.stds[ii]
            if std0 == 0:
                std0 = 1
            df[col0] = (df[col0] - mean0)/std0
        return df

    def fit_transform(self,df):
        self.fit(df)
        self.transform(df)
        return df
    
    def inverse_transform(self,df):
        for ii in range(0,len(self.columns)):
            col0 = self.columns[ii]
            mean0 = self.means[ii]
            std0 = self.stds[ii]
            if std0 == 0:
                std = 1
            df[col0] = (df[col0] * std0) + mean0
        return df
