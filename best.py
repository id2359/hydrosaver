
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
from sklearn.preprocessing.imputation import Imputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

RUN_NAME = "r1"
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
                                                        test_size=0.50,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


# In[4]:

def get_features(df):
    #return df[COLUMNS_TO_INCLUDE].copy()
    return df[df.columns.difference([TARGET_COLUMN]+COLUMNS_TO_EXCLUDE)].copy()

def get_target(df):
    return df[[TARGET_COLUMN]]


# In[5]:

TRAIN_CSV = '/home/frug/projects/hydrosaver/train.csv'


# In[6]:

TEST_CSV = '/home/frug/projects/hydrosaver/test.csv'


# In[7]:

train = load_csv(TRAIN_CSV)


# In[8]:

x = get_features(train)


# In[9]:

y = get_target(train)


# In[10]:

x_train, x_test, y_train, y_test = ttsplit(x,y)


# In[11]:

imputer = Imputer(copy=False, verbose=True)
scaler = StandardScaler()


# In[12]:

param_grid = {"n_estimators": [200],
    "max_depth": [25],
    "max_features": [12]}

rfr = RandomForestRegressor(random_state=0)

grid = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=-1, cv=5)


# In[13]:

pipeline = make_pipeline(imputer, scaler,grid)


# In[14]:

model = pipeline.fit(x_train, y_train)


# In[15]:

grid.best_score_


# In[16]:

grid.best_params_


# In[17]:

grid.best_estimator_


# In[18]:

predictions = model.predict(x_test)


# In[19]:

np.sqrt(mean_squared_error(predictions, y_test))


# In[20]:

mean_absolute_error(predictions, y_test)


# In[21]:

model.score(x_test, y_test)


# In[22]:

real_data = load_csv(TEST_CSV)


# In[23]:

real_features = get_features(real_data)


# In[24]:

real_predictions = model.predict(real_features)


# In[25]:

np.savetxt('/home/frug/predictions_2_%s.csv' % RUN_NAME, real_predictions,"%.4f")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



