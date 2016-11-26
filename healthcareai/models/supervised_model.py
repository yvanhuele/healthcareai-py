import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from healthcareai.common.connections import write_scores_to_sql
from healthcareai.common.model_eval import get_top_k_features

class SupervisedModel(object):
    def __init__(self,
                 model,
                 feature_model,
                 pipeline,
                 column_names,
                 predictiontype,
                 graincol,
                 y_pred,
                 y_actual):
        """
        A trained regressor or classifier
        :param model:
        :param feature_model:
        :param pipeline:
        :param column_names:
        :param predictiontype:
        :param graincol:
        :param y_pred:
        :param y_actual:
        """
        self.model = model
        self.feature_model = feature_model
        self.pipeline = pipeline
        self.column_names = column_names
        self.predictiontype = predictiontype
        self.graincol = graincol
        self.y_pred = y_pred
        self.y_actual = y_actual

    def save(self, filepath):
        pickle.dump(self, open(filepath, 'wb'))

    def score(self, df_to_score, saveto=None, numtopfeatures=3):
        """
        Returns model with predicted probability scores and top 3 features
        :param df_to_score: the data to be scored
        :param saveto: 'sql' or a filepath. 'sql' directs the save to
            the SAM database as per the documentsation. A filepath saves
            a csv of the data, but with an additional column of scores.
        :returns: the input database with a column of scores.
        """

        # Get predictive scores
        df = df_to_score.copy()
        df = self.pipeline.transform(df)
        df = df[[c for c in df.columns if c in self.column_names]]

        y_pred = self.model.predict_proba(df)[:, 1]

        # Get top 3 reasons
        reason_col_names = ['Factor%iTXT'%i for i in range(1,numtopfeatures+1)]
        top_feats_lists = get_top_k_features(df, self.feature_model,
                                             k=numtopfeatures)

        # join prediction and top features columns to dataframe
        df['y_pred'] = y_pred
        reasons_df = pd.DataFrame(top_feats_lists, columns=reason_col_names,
                                  index=df.index)
        df = pd.concat([df, reasons_df], axis=1, join_axes=[df.index])

        # bring back the grain column and reset the df index
        df.insert(0, self.graincol, df_to_score[self.graincol])
        df.reset_index(drop=True, inplace=True)

        if saveto == 'sql':
            write_scores_to_sql(df,
                                predictiontype=self.predictiontype,
                                graincol=self.graincol,
                                )
        elif saveto != None:
            df.to_csv(saveto)

        return df

    def get_roc_auc(self):
        """
        Returns the roc_auc of the holdout set from model training.
        """
        return roc_auc_score(self.y_actual, self.y_pred)

    def roc_curve_plot(self):
        """
        Returns a plot of the roc curve of the holdout set from model training.
        """
        pass