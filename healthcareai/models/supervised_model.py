from sklearn.metrics import roc_auc_score
from healthcareai.common.connections import write_scores_to_sql

class SupervisedModel(object):
    def __init__(self,
                 model,
                 pipeline,
                 column_names,
                 predictiontype,
                 graincol,
                 y_pred,
                 y_actual):
        """
        Docstring here
        """
        self.model = model
        self.pipeline = pipeline
        self.column_names = column_names
        self.predictiontype = predictiontype
        self.graincol = graincol
        self.y_pred = y_pred
        self.y_actual = y_actual

    def score(self, df_to_score, saveto=None):
        """
        Returns model with predicted probability scores in
        :param score_df: the data to be scored
        :param save: 'sql' or a filepath. 'sql' directs the save to
            the SAM database as per the documentation. A filepath saves
            a csv of the data, but with an additional column of scores.
        :returns: the input database with a column of scores.
        """

        # Get predictive scores
        df = df_to_score.copy()
        df = self.pipeline.transform(df)
        df = df[[c for c in df.columns if c in self.column_names]]
        df['y_pred'] = self.model.predict_proba(df)[:, 1]
        df[self.graincol] = df_to_score[self.graincol]

        # Get top 3 reasons
        # TODO calculate top 3 factors
        df['Factor1TXT'] = 'Thing1'
        df['Factor2TXT'] = 'Thing2'
        df['Factor3TXT'] = 'Thing3'

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