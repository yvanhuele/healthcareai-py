import unittest
import pandas as pd
from healthcareai.common.functions import ImpactCoding


    # def test_column_renaming(self):
    #     df = pd.read_csv(fixture('iris_classification.csv'), na_values=['None'])
    #     code_column_name = 'DRG'
    #     test_df = impact_coding_on_a_single_column(df, 'species', code_column_name)

    #     self.assertTrue((code_column_name + '_impact_coded') in test_df.columns)

    # def test_unique_values(self):
    #     df = pd.read_csv(fixture('iris_classification.csv'), na_values=['None'])
    #     unique_drgs = len(df.DRG.unique())
    #     test_df = impact_coding_on_a_single_column(df, 'species', 'DRG')
    #     unique_impact_values = len(test_df.DRG_impact_coded.unique())

    #     self.assertLessEqual(unique_impact_values, unique_drgs)


class TestClassImpactCoding(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            ['a', 'a', 2],
            ['b', 'a', 1],
            ['b', 'b', 2]
        ])
        self.df.columns = ['one','two','three']
        
    def test_number_of_unique_values_in_impact_coded_columns(self):
        ic = ImpactCoding(['one','two'],'three')
        df_final = ic.fit_transform(self.df)
        self.assertEqual(len(df_final.one.unique()), 2)
        self.assertEqual(len(df_final.two.unique()), 2)

        
    def tearDown(self):
        del self.df


if __name__ == '__main__':
    unittest.main()
    
