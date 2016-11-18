from healthcareai.common.filters import DateTimeColumnFilter, GrainColumnFilter, NullValueFilter
from healthcareai.common.transformers import DataFrameImputer, Dummifier
from healthcareai.models.supervised_model import SupervisedModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import pickle

class SupervisedModelTrainer(object):
    def __init__(self,
                 predictiontype,
                 modeltype,
                 predictedcol,
                 graincol=None,
                 impute=True):
        """
        Configures a Supervised Model Trainer object
        :param predictiontype ['classification', 'regression']
        :param modeltype ['rf', 'linear']
        :param graincol
        :param windowcol
        :param impute: default=True. Determines whether the trainer
            will impute NaN values.  If not, will drop rows.
        :
        """
        self.predictiontype = predictiontype
        self.modeltype = modeltype
        self.predictedcol = predictedcol
        self.impute = impute
        self.graincol = graincol

    def train(self, df, savepath=None):
        """
        Trains and returns a supervised model
        :param df training data
        :returns an instance of SupervisedModel class
        """

        #Preprocess data
        pipeline = Pipeline([
            ('dts_filter', DateTimeColumnFilter()),
            ('graincol_filter', GrainColumnFilter(self.graincol)),
            ('imputer', DataFrameImputer(impute=self.impute)),
            ('dropna', NullValueFilter(ignorecols=[self.predictedcol])),
            ('dummify', Dummifier(ignorecols=[self.predictedcol])),
        ])

        df = pipeline.fit_transform(df)
        if 'Y' in df[self.predictedcol].unique():
            df[self.predictedcol] = (df[self.predictedcol] == 'Y').astype(int)

        X = df.drop(self.predictedcol, axis=1)
        y = df[self.predictedcol].values

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2)

        #Determine model type
        if self.predictiontype == 'classification' and self.modeltype == 'rf':
            model = RandomForestClassifier()

        elif self.predictiontype == 'classification' and self.modeltype == 'linear':
            model = LogisticRegression()

        elif self.predictiontype == 'regression' and self.modeltype == 'rf':
            model = RandomForestRegressor()

        elif self.predictiontype == 'regression' and self.modeltype == 'linear':
            model = LinearRegression()

        #Train model
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:,1]
        column_names = X_test.columns.values
        finalmodel = SupervisedModel(model,
                                     pipeline,
                                     column_names,
                                     self.predictiontype,
                                     y_pred,
                                     y_test)

        #Save if applicable
        if savepath != None:
            pickle.dump(finalmodel, open(savepath, 'wb'))

        return finalmodel

