
# coding: utf-8

# In[18]:

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing.imputation import Imputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def ttsplit(x,y):
    # split into a training and testing set
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test
    

imputer = Imputer(copy=False, verbose=True)
scaler = StandardScaler()


# In[2]:

def to_float(df):
    return df.apply(pd.to_numeric, errors='coerce')

def fillna(df):
    return df.fillna(df.mean())

def load_csv(filename):
    return fillna(to_float(pd.read_csv(filename, header=0, low_memory=False)))


# In[3]:

TARGET_COLUMN = 'target'
COLUMNS_TO_EXCLUDE = ['timestamp',
                      'DIC88023.PV',
                      'II88151.PV',
                      'II88152.PV',
                      'FV88156.PV',
                      'FV88043.PV',
                      'FV88044.PV']

def get_features(df):
    return df[df.columns.difference([TARGET_COLUMN]+COLUMNS_TO_EXCLUDE)].copy()

def get_target(df):
    return df[[TARGET_COLUMN]]


# In[4]:

lm = linear_model.LinearRegression()


# In[5]:

# TRAIN_CSV = '/home/frug/projects/hydrosaver/train.csv'
TRAIN_CSV = '/media/sf_ncm/train.csv'


# In[6]:

# TEST_CSV = '/home/frug/projects/hydrosaver/test.csv'
TEST_CSV = '/media/sf_ncm/test.csv'


# In[7]:

train = load_csv(TRAIN_CSV)


# In[8]:

x = get_features(train)


# In[9]:

y = get_target(train)


# In[10]:

x_train, x_test, y_train, y_test = ttsplit(x,y)


# In[11]:

pipeline = make_pipeline(imputer,scaler, lm)


# In[12]:

model = pipeline.fit(x_train, y_train)


# In[13]:

predictions_test = model.predict(x_test)


# In[14]:

predictions_test


# In[17]:

mean_squared_error(predictions_test, y_test)


# In[19]:

mean_absolute_error(predictions_test, y_test)


# In[ ]:



