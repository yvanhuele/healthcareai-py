import pandas as pd


def top_cols(row):
    """
    Sorts (descending) the columns of a dataframe by value or a single row
    :param row: a row of a pandas data frame
    :return: an array of column names
    """
    return row.sort_values(ascending=False).index.values

def get_top_k_features(df, linear_model, k=3):
    """
    Get lists of top features based on an already-fit linear model
    :param df: The dataframe for which to score top features
    :param linear_model: A pre-fit scikit learn model instance that has linear
        coefficients.
    :return: k lists of top features (the first list is the top features, the
        second list are the #2 features, etc)
    """
    res = pd.DataFrame(df.values * linear_model.coef_, columns=df.columns)
    res.apply(top_cols, axis=1)
    return list(res.apply(top_cols, axis=1).values[:, :k])

if __name__ == "__main__":
    pass
