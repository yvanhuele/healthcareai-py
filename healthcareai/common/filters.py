import pandas as pd
from sklearn.preprocessing import StandardScaler

def remove_datetime_columns(df):
    # TODO: make this work with col names shorter than three letters
    cols = [c for c in df.columns if c[-3:] != 'DTS']
    return df[cols]

def feature_scaling(X_train,X_test,columns_to_scale):
    X_train_sub = X_train[columns_to_scale]
    X_test_sub = X_test[columns_to_scale]        
    scaler = StandardScaler()
    scaler.fit(X_train_sub)
    
    xx = pd.DataFrame(scaler.transform(X_train_sub))
    xx.index = X_train_sub.index
    xx.columns = X_train_sub.columns
    X_train[columns_to_scale] = xx
    
    xx = pd.DataFrame(scaler.transform(X_test_sub))
    xx.index = X_test_sub.index
    xx.columns = X_test_sub.columns
    X_test[columns_to_scale] = xx

    return X_train, X_test
