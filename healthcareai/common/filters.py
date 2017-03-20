import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def remove_datetime_columns(df):
    # TODO: make this work with col names shorter than three letters
    cols = [c for c in df.columns if c[-3:] != 'DTS']
    return df[cols]

def undersampling(X,y):
    US = RandomUnderSampler(random_state=1)
    usx, usy = US.fit_sample(X,y)
    
    usx = pd.DataFrame(usx)
    usx.columns = X.columns
    usy = pd.Series(usy)

    return usx, usy

def oversampling(X,y):
    OS = RandomOverSampler(random_state=1)
    osx, osy = OS.fit_sample(X,y)
    
    osx = pd.DataFrame(osx)
    osx.columns = X.columns
    osy = pd.Series(osy)

    return osx, osy
