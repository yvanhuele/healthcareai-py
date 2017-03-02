import os
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from healthcareai.common import model_eval
from healthcareai.common.transformers import DataFrameImputer
from healthcareai.common import filters


class DevelopSupervisedModel(object):
    """
    This class helps create a model using several common classifiers
    (reporting AUC) and regressors (reporting MAE/MSE). When instantiating,
    the data is prepped and one-fifth is held out so model accuracy can be
    assessed.

    Parameters
    ----------
    modeltype (str) : whether the model will be 'classification' or
    'regression'

    dataframe (dataframe) : data that your model is based on

    predictedcol (str) : y column (in ticks) who's values are being predicted

    impute (boolean) : whether imputation is done on the data; if not,
    rows with nulls are removed

    graincol (str) : OPTIONAL | column (in ticks) that represents the data's
    grain

    debug (boolean) : OPTIONAL | verbosity of the output

    Returns
    -------
    Object representing the cleaned data, against which methods are run
    """

    def __init__(self, modeltype, dataframe, predictedcol, impute, graincol=None, debug=False):

        self.modeltype = modeltype
        self.dataframe = dataframe
        self.predictedcol = predictedcol
        self.impute = impute
        self.y_probab_linear = None
        self.y_probab_rf = None
        self.col_list = None
        self.rfclf = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.au_roc = None

        if debug:
            print('Shape and top 5 rows of original dataframe:')
            print(self.dataframe.shape)
            print(self.dataframe.head())

        # remove datetime columns
        self.dataframe = filters.remove_datetime_columns(self.dataframe)

        # Remove graincol (if specified)
        if graincol:
            self.dataframe.drop(graincol, axis=1, inplace=True)

        if debug:
            print('\nDataframe after removing DTS columns:')
            print(self.dataframe.head())
            print('\nNow either doing imputation or dropping rows with NULLs')

        if self.impute:
            self.dataframe = DataFrameImputer().fit_transform(self.dataframe)
            # This class comes from here:
            # http://stackoverflow.com/a/25562948/5636012
            if debug:
                print('\nself.df after doing imputation:')
                print(self.dataframe.shape)
                print(self.dataframe.head())
        else:
            # TODO switch similar statements to work inplace
            self.dataframe = self.dataframe.dropna(axis=0, how='any', inplace=True)
            print('\nself.df after dropping rows with NULLS:')
            print(self.dataframe.shape)
            print(self.dataframe.head())

        # CALL new function!!
        # Convert predicted col to 0/1 (otherwise won't work with GridSearchCV)
        # Note that this makes healthcareai only handle N/Y in pred column
        if self.modeltype == 'classification':
            # Turning off warning around replace
            pd.options.mode.chained_assignment = None  # default='warn'
            # TODO: put try/catch here when type = class and predictor is numer
            self.dataframe[self.predictedcol].replace(['Y', 'N'], [1, 0],
                                                      inplace=True)

            if debug:
                print('\nDataframe after converting to 1/0 instead of Y/N for '
                      'classification:')
                print(self.dataframe.head())

        # Remove rows with null values in predicted col
        self.dataframe = self.dataframe[pd.notnull(self.dataframe[self.predictedcol])]

        if debug:
            print('\nself.df after removing rows where predicted col is NULL:')
            print(self.dataframe.shape)
            print(self.dataframe.head())

        # Create dummy vars for all cols but predictedcol
        # First switch (temporarily) pred col to numeric (so it's not dummy)
        self.dataframe[self.predictedcol] = pd.to_numeric(
            arg=self.dataframe[self.predictedcol], errors='raise')
        self.dataframe = pd.get_dummies(self.dataframe, drop_first=True, prefix_sep='.')

        y = np.squeeze(self.dataframe[[self.predictedcol]])
        X = self.dataframe.drop([self.predictedcol], axis=1)

        # Split the dataset in two equal parts
        self.X_train, self.X_test, self.y_train, self.y_test = \
            model_selection.train_test_split(
                X, y, test_size=.20, random_state=0)

        if debug:
            print('\nShape of X_train, y_train, X_test, and y_test:')
            print(self.X_train.shape)
            print(self.y_train.shape)
            print(self.X_test.shape)
            print(self.y_test.shape)

    def linear(self, cores=4, debug=False):
        """
        This method creates and assesses the accuracy of a logistic regression
        model.

        Parameters
        ----------
        cores (num) : Number of cores to use (default 4)
        debug (boolean) : Verbosity of output (default False)

        Returns
        -------
        Nothing. Output to console describes model accuracy.
        """

        if self.modeltype == 'classification':
            algo = LogisticRegressionCV(cv=5)

        # TODO: get GroupLasso working via lightning

        # TODO: see if CV splits needed for linear regress

        elif self.modeltype == 'regression':
            algo = LinearRegression()
        else:
            algo = None

        self.y_probab_linear, self.au_roc = model_eval.clfreport(modeltype=self.modeltype, debug=debug,
                                                                 devcheck='yesdev', algo=algo, X_train=self.X_train,
                                                                 y_train=self.y_train, X_test=self.X_test,
                                                                 y_test=self.y_test, cores=cores)

    def random_forest(self, cores=4, trees=200, tune=False, debug=False):
        """
        This method creates and assesses the accuracy of a logistic regression
        model.

        Parameters
        ----------
        cores (num) : Number of cores to use (default 4)
        trees (num) : Number of trees in the random forest (default 200)
        tune (boolean) : Whether to tune hyperparameters. This iterates number
        of trees from 100, 250, and 500.
        debug (boolean) : Verbosity of output (default False)

        Returns
        -------
        Nothing. Output to console describes model accuracy.
        """

        # TODO: refactor, such that each algo doesn't need an if/else tree
        if self.modeltype == 'classification':
            algo = RandomForestClassifier(n_estimators=trees,
                                          verbose=(2 if debug is True else 0))

        elif self.modeltype == 'regression':
            algo = RandomForestRegressor(n_estimators=trees,
                                         verbose=(2 if debug is True else 0))

        else:  # Here to appease pep8
            algo = None

        params = {'max_features':
                      model_eval.calculate_rfmtry(len(self.X_test.columns),
                                                  self.modeltype)}

        self.col_list = self.X_train.columns.values

        self.y_probab_rf, self.au_roc, self.rfclf = model_eval.clfreport(modeltype=self.modeltype, debug=debug,
                                                                         devcheck='yesdev', algo=algo,
                                                                         X_train=self.X_train, y_train=self.y_train,
                                                                         X_test=self.X_test, y_test=self.y_test,
                                                                         param=params, cores=cores, tune=tune,
                                                                         col_list=self.col_list)

    def plot_roc(self, filename=None, debug=False):
        """
        Plots roc related to models resulting from linear and random
        forest methods within the DevelopSupervisedModel step.

        Parameters
        ----------
        filename (string) : Optional filename to save plot as
        debug (boolean) : Verbosity of output. If True, shows list of FPR/TPR for each point in the plot (default False)

        Returns
        -------
        Nothing. A plot is created and displayed.
        """

        fpr_linear, tpr_linear, _ = roc_curve(self.y_test,
                                              self.y_probab_linear)
        roc_auc_linear = auc(fpr_linear, tpr_linear)

        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, self.y_probab_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)

        # TODO: add cutoff associated with FPR/TPR
        if debug:
            print('Linear model:')
            print('FPR, and TRP')
            print(pd.DataFrame(
                {'FPR': fpr_linear, 'TPR': tpr_linear}))

            print('Random forest model:')
            print('FPR, and TRP')
            print(pd.DataFrame({'FPR': fpr_rf, 'TPR': tpr_rf}))

        plt.figure()
        plt.plot(fpr_linear, tpr_linear, color='b',
                 label='Logistic (area = %0.2f)' % roc_auc_linear)
        plt.plot(fpr_rf, tpr_rf, color='g',
                 label='RandomForest (area = %0.2f)' % roc_auc_rf)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        if filename:
            self._save_plot_as_png(plt, filename)

        plt.show()

    def plot_rffeature_importance(self, filename=None):
        """
        Plots feature importances related to models resulting from
        and random forest methods within the DevelopSupervisedModel step.

        Parameters
        ----------
        filename (string) : If this is passed in, saves the plot as that filename.png

        Returns
        -------
        Nothing. A plot is created and displayed.
        """

        # Arrange columns in order of importance
        if hasattr(self.rfclf, 'best_estimator_'):
            importances = self.rfclf.best_estimator_.feature_importances_
            std = np.std([tree.feature_importances_ for tree in self.rfclf.best_estimator_.estimators_], axis=0)
        else:
            importances = self.rfclf.feature_importances_
            std = np.std([tree.feature_importances_ for tree in self.rfclf.estimators_], axis=0)

        indices = np.argsort(importances)[::-1]
        namelist = [self.col_list[i] for i in indices]

        # Plot these columns
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        plt.xticks(range(self.X_train.shape[1]), namelist, rotation=90)
        plt.xlim([-1, self.X_train.shape[1]])
        plt.gca().set_ylim(bottom=0)
        plt.tight_layout()

        if filename:
            self._save_plot_as_png(plt, filename)

        plt.show()

    @staticmethod
    def _save_plot_as_png(plot, filename):
        """
        Saves a plot as a .png with the given filename
        :param plot: matplotlib.pyplot
        :param filename: a string filename
        """
        plot.savefig('{}.png'.format(filename))
        source_path = os.path.dirname(os.path.abspath(__file__))
        print('Plot saved in: {}/{}'.format(source_path, filename))
