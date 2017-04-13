import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from datetime import datetime
import json
import pickle

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
        return df

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
        
        
def impact_coding(dataframe,
                  predicted_column,
                  impact_column,
                  random_state = 0):
    train, test = train_test_split(dataframe,
                                   test_size=0.8,
				   random_state = random_state)
    train = train.copy()
    test = test.copy()
    x_bar = train[predicted_column].mean()
    dic = {}
    for category in train[impact_column].unique():
        key0 = category
        sub_dataframe = train[train[impact_column] == category]
        value0 = sub_dataframe[predicted_column].mean()
        dic[key0] = value0
    test[impact_column] = np.where(test[impact_column].isin(dic.keys()),
                                   test[impact_column].replace(dic),
				   x_bar) \
                                   - x_bar
    return test

def impact_coding_learn(dataframe,
                  predicted_column,
                  impact_column,
                  random_state = 0):
    train, test = train_test_split(dataframe,
                                   test_size=0.8,
				   random_state = random_state)
    train = train.copy()
    test = test.copy()
    x_bar = train[predicted_column].mean()
    dic = {}
    for category in train[impact_column].unique():
        key0 = category
        sub_dataframe = train[train[impact_column] == category]
        value0 = sub_dataframe[predicted_column].mean()
        dic[key0] = value0
    return test, dic, x_bar

def impact_coding_transform(dataframe,impact_column,dic,x_bar):
    dataframe[impact_column] = np.where(
                                   dataframe[impact_column].isin(dic.keys()),
                                   dataframe[impact_column].replace(dic),
				   x_bar) \
                                   - x_bar
    return dataframe

def under_sampling(df,predicted_column,random_state=0):
    y = df[predicted_column]
    X = df.drop(predicted_column, axis=1)
    
    under_sampler = RandomUnderSampler(random_state=random_state)
    X_under_sampled, y_under_sampled = under_sampler.fit_sample(X,y)
    
    X_under_sampled = pd.DataFrame(X_under_sampled)
    X_under_sampled.columns = X.columns
    y_under_sampled = pd.Series(y_under_sampled)
    
    dataframe_under_sampled = X_under_sampled
    dataframe_under_sampled[predicted_column] = y_under_sampled 
    return dataframe_under_sampled

def over_sampling(df,predicted_column,random_state=0):
    y = df[predicted_column]
    X = df.drop(predicted_column, axis=1)

    over_sampler = RandomOverSampler(random_state=random_state)
    X_over_sampled, y_over_sampled = over_sampler.fit_sample(X,y)
    
    X_over_sampled = pd.DataFrame(X_over_sampled)
    X_over_sampled.columns = X.columns
    y_over_sampled = pd.Series(y_over_sampled)
    
    dataframe_over_sampled = X_over_sampled
    dataframe_over_sampled[predicted_column] = y_over_sampled 
    return dataframe_over_sampled

def train_test_splt(df,predicted_column,
                    train_size = 0.8, random_state = 0):
    y = df[predicted_column]
    X = df.drop(predicted_column, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size = train_size, random_state = random_state)
    return X_train, X_test, y_train, y_test

def feature_scaling(X_train, X_test, columns_to_scale):
    X_train_scaled_subset = X_train[columns_to_scale]
    X_test_scaled_subset = X_test[columns_to_scale]        
    scaler = StandardScaler()
    scaler.fit(X_train_scaled_subset)
    
    X_train_scaled_subset_dataframe = \
        pd.DataFrame(scaler.transform(X_train_scaled_subset))
    X_train_scaled_subset_dataframe.index =  X_train_scaled_subset.index
    X_train_scaled_subset_dataframe.columns = X_train_scaled_subset.columns
    X_train[columns_to_scale] = X_train_scaled_subset_dataframe
    
    X_test_scaled_subset_dataframe = \
        pd.DataFrame(scaler.transform(X_test_scaled_subset))
    X_test_scaled_subset_dataframe.index = X_test_scaled_subset.index
    X_test_scaled_subset_dataframe.columns = X_test_scaled_subset.columns
    X_test[columns_to_scale] = X_test_scaled_subset_dataframe

    return X_train, X_test

def randomsearch(X,y,input_grids,
                 cv=5,scoring_metric='roc_auc',random_state=0):
    estimator_dictionary = {}    
    print('\n\n')
    print('**************************************************')
    print('Cross-Validation performance of models on TRAIN SET:')
    print('**************************************************')
    for ii in range(0,len(input_grids)):
        grid = RandomizedSearchCV(estimator =
                                  eval(input_grids['estimator'][ii]),
                                  param_distributions =
                                  eval(input_grids['param_grid'][ii]),
                                  cv = cv,
                                  scoring = scoring_metric,
                                  n_iter = input_grids['n_iter'][ii],
                                  random_state = random_state)
        grid.fit(X,y)
        print(   input_grids['estimator'][ii][0:-2]
                 + " score: "
                 + str(grid.best_score_)   )
        estimator_dictionary[input_grids['estimator'][ii][0:-2]] = \
                                                    grid.best_estimator_
    return estimator_dictionary
            
def choose_best_model(X,y,estimator_dictionary):
    print('\n\n')
    print('**************************************************')
    print('Performance of models on TEST SET:')
    print('**************************************************')
    best_score = 0
#    for estimator0 in list(estimator_dictionary.values()):
    for key0 in list(estimator_dictionary.keys()):
        estimator0 = estimator_dictionary[key0]
        y_pred = estimator0.predict(X)
        roc_auc = metrics.roc_auc_score(y_true = y, y_score = y_pred)
        print(key0 + ' score: ' + str(roc_auc))
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = estimator0
    print("Best estimator: ")
    print(best_model)
    print("Best score on test set:")
    print(best_score)
    return best_model
        
def write_classification_metrics_to_json(y_real,y_pred,file_name = ''):
    output = {}
    accuracy = metrics.accuracy_score(y_real,y_pred)
    confusion_matrix = metrics.confusion_matrix(y_real,y_pred)
    output['accuracy'] = accuracy
    output['confusion_matrix'] = confusion_matrix.tolist()
    fmt = '%Y-%m-%d_%H-%M-%S'
    file_name = file_name + "_"  \
                + str(datetime.utcnow().strftime(fmt)) \
                + ".json"
    with open(file_name, 'w') as fp:
        json.dump(output, fp, indent=4, sort_keys=True)
        
def write_model_to_pickle(model,file_name = ''):
    fmt = '%Y-%m-%d_%H-%M-%S'
    file_name = file_name + "_"  \
                + str(datetime.utcnow().strftime(fmt)) \
                + ".pkl"
    with open(file_name, "wb") as output_file:
        pickle.dump(model, output_file)
