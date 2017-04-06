import pandas as pd
import numpy as np
import healthcareai as hc

# Make dataframe with NULLS and a column called DRG to do impact coding on.

df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
df = df.head(60)
df['species'].replace(['setosa','versicolor'],[0,1],inplace=True)
DRG = []
for ii in range(0,df.shape[0]):
    if ii < 40:
        DRG.append('verde')
    elif ii >=40 and ii <= 55 :
        DRG.append('azul')
    else:
        DRG.append('amarillo')
DRG = pd.Series(DRG)
df['DRG'] = DRG
df = pd.concat([df,df])
df.index = range(0,120)
df.iloc[[10,20,30,40,50,60,70,80,90,100,110],[0,1,2,3,5]] = None


oo = hc.DevelopSupervisedModel(dataframe = df,
                               predicted_column = 'species',
                               model_type = 'classification')


oo.imputation()
oo.impact_coding_on_a_single_column(column = 'DRG')
oo.over_sampling()
oo.train_test_split()
oo.feature_scaling(['sepal_length','sepal_width','petal_length','petal_width','DRG'])

results = oo.ensemble_classification()
