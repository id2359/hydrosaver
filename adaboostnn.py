
# coding: utf-8

# In[134]:

import pandas as pd
import numpy as np
from sklearn.preprocessing.imputation import Imputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor

RUN_NAME = "nntest21211000_20"
TARGET_COLUMN = 'target'
COLUMNS_TO_EXCLUDE = ['timestamp',
                      'DIC88023.PV',
                      'II88151.PV',
                      'II88152.PV',
                      'FV88156.PV',
                      'FV88043.PV',
                      'FV88044.PV']


def to_float(df):
    return df.apply(pd.to_numeric, errors='coerce')

def fillna(df):
    return df.fillna(df.mean())

def load_csv(filename):
    return fillna(to_float(pd.read_csv(filename, header=0)))

def ttsplit(x,y):
    # split into a training and testing set
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


# In[135]:

def get_features(df):
    #return df[COLUMNS_TO_INCLUDE].copy()
    return df[df.columns.difference([TARGET_COLUMN]+COLUMNS_TO_EXCLUDE)].copy()

def get_target(df):
    return df[[TARGET_COLUMN]]


# In[136]:

imputer = Imputer(copy=False, verbose=True)
scaler = StandardScaler()


# In[137]:

TRAIN_CSV = '/home/frug/projects/hydrosaver/train.csv'


# In[138]:

TEST_CSV = '/home/frug/projects/hydrosaver/test.csv'


# In[139]:

train = load_csv(TRAIN_CSV)


# In[140]:

x = get_features(train)


# In[141]:

len(x.columns)


# In[142]:

y = get_target(train)


# In[143]:

mlp = MLPRegressor(hidden_layer_sizes=(21,21,), max_iter = 1000, learning_rate = 'adaptive')


# In[144]:

x_train, x_test, y_train, y_test = ttsplit(x,y)


# In[145]:

ada = AdaBoostRegressor(mlp, random_state=1, n_estimators=40)


# In[146]:

pipeline = make_pipeline(imputer, scaler, mlp)


# In[147]:

model = pipeline.fit(x_train, y_train)


# In[148]:

predictions = model.predict(x_test)


# In[149]:

np.sqrt(mean_squared_error(predictions, y_test))


# In[150]:

mean_absolute_error(predictions, y_test)


# In[151]:

model.score(x_test, y_test)


# In[152]:

real_data = load_csv(TEST_CSV)


# In[153]:

real_features = get_features(real_data)


# In[154]:

real_predictions = model.predict(real_features)


# In[155]:

np.savetxt('/home/frug/%s_predictions.csv' % RUN_NAME, real_predictions,"%.4f")


# In[ ]:



