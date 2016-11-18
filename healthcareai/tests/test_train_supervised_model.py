import unittest

import numpy as np
import pandas as pd
from healthcareai.tests.helpers import fixture
from healthcareai import SupervisedModelTrainer

class TestTrainRF(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])

        # Drop uninformative columns
        df.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

        # Convert numeric columns to factor/category columns
        np.random.seed(42)
        train_params = {
            'predictiontype': 'classification',
            'predictedcol': 'ThirtyDayReadmitFLG',
            'graincol': 'PatientID',
        }
        self.trainer = SupervisedModelTrainer(modeltype='rf', **train_params)
        self.model = self.trainer.train(df)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.model.get_roc_auc(), 6), 0.942842)

    def tearDown(self):
        del self.model
        del self.trainer


class TestTrainLogistic(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])

        # Drop uninformative columns
        df.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

        # Convert numeric columns to factor/category columns
        np.random.seed(42)
        train_params = {
            'predictiontype': 'classification',
            'predictedcol': 'ThirtyDayReadmitFLG',
            'graincol': 'PatientID',
        }
        self.trainer = SupervisedModelTrainer(modeltype='linear', **train_params)
        self.model = self.trainer.train(df)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.model.get_roc_auc(), 6), 0.703684)

    def tearDown(self):
        del self.model
        del self.trainer