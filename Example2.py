"""
Important: Make sure to run Example1.py first, or otherwise
save a model under the filepath 'rf_model.pkl'

Example2.py shows how to get predictive scores from a saved model,
and save them.
"""
from healthcareai.common.connections import load_saved_model
import pandas as pd
import time


def main():

    t0 = time.time()

    # load data to score
    # we will use the set defined by InTestWindow in our .csv

    df = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv',
                     na_values=['None'])
    df_to_score = df[df['InTestWindowFLG'] == 'Y']

    rf_model = load_saved_model('rf_model.pkl')

    # Score records and save scores to .csv.
    # To save instead to a SAM database, use saveto='sql'
    # If you do not need to save scores, leave out the saveto parameter.

    df_scored = rf_model.score(df_to_score, saveto='scores.csv')
    print('scored records:\n', df_scored.head())

    print('\nTime:\n', time.time() - t0)

if __name__ == "__main__":
    main()
