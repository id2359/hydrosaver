
# coding: utf-8

# In[83]:

import pandas as pd
import numpy as np
from sklearn.preprocessing.imputation import Imputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

RUN_NAME = "rfrgscv"
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


# In[84]:

def get_features(df):
    #return df[COLUMNS_TO_INCLUDE].copy()
    return df[df.columns.difference([TARGET_COLUMN]+COLUMNS_TO_EXCLUDE)].copy()

def get_target(df):
    return df[[TARGET_COLUMN]]


# In[85]:

TRAIN_CSV = '/home/frug/projects/hydrosaver/train.csv'


# In[86]:

TEST_CSV = '/home/frug/projects/hydrosaver/test.csv'


# In[87]:

train = load_csv(TRAIN_CSV)


# In[88]:

x = get_features(train)


# In[89]:

y = get_target(train)


# In[90]:

x_train, x_test, y_train, y_test = ttsplit(x,y)


# In[91]:

imputer = Imputer(copy=False, verbose=True)
scaler = StandardScaler()


# In[92]:

param_grid = {"n_estimators": [60,80,100],
    "max_depth": [7,8,9,20],
    "max_features": [12,15,20]}

rfr = RandomForestRegressor(random_state=0)
grid = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=-1)


# In[93]:

pipeline = make_pipeline(imputer, scaler,grid)


# In[94]:

model = pipeline.fit(x_train, y_train)


# In[95]:

grid.best_score_


# In[96]:

grid.best_params_


# In[97]:

grid.best_estimator_


# In[98]:

predictions = model.predict(x_test)


# In[99]:

np.sqrt(mean_squared_error(predictions, y_test))


# In[100]:

mean_absolute_error(predictions, y_test)


# In[101]:

model.score(x_test, y_test)


# In[102]:

real_data = load_csv(TEST_CSV)


# In[103]:

real_features = get_features(real_data)


# In[104]:

real_predictions = model.predict(real_features)


# In[105]:

np.savetxt('/home/frug/predictions_%s.csv' % RUN_NAME, real_predictions,"%.4f")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



