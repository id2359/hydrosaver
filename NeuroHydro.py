
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from sklearn.preprocessing.imputation import Imputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor

def ttsplit(x,y):
    # split into a training and testing set
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test
    

imputer = Imputer(copy=False, verbose=True)
scaler = StandardScaler()
feature_selector = VarianceThreshold(threshold=(.8 * (1 - .8)))


# In[ ]:

def to_float(df):
    return df.apply(pd.to_numeric, errors='coerce')

def fillna(df):
    return df.fillna(df.mean())

def load_csv(filename):
    return fillna(to_float(pd.read_csv(filename, header=0)))


# In[ ]:

TARGET_COLUMN = 'target'
COLUMNS_TO_EXCLUDE = ['timestamp',
                      'DIC88023.PV',
                      'II88151.PV',
                      'II88152.PV',
                      'FV88156.PV',
                      'FV88043.PV',
                      'FV88044.PV']

#COLUMNS_TO_INCLUDE = ['FX87211.CPV1',       
#                      'MQI88024.CPV',  
#                      'WQI8100XCL1.CPV']

def get_features(df):
    #return df[COLUMNS_TO_INCLUDE].copy()
    return df[df.columns.difference([TARGET_COLUMN]+COLUMNS_TO_EXCLUDE)].copy()

def get_target(df):
    return df[[TARGET_COLUMN]]


# In[ ]:

mlp = MLPRegressor(hidden_layer_sizes=(5,5,5), max_iter = 100, solver='lbfgs',                    alpha=0.01, activation = 'tanh', random_state = 8)


# In[ ]:

TRAIN_CSV = '/home/frug/projects/hydrosaver/train.csv'


# In[ ]:

TEST_CSV = '/home/frug/projects/hydrosaver/test.csv'


# In[ ]:




# In[ ]:

train = load_csv(TRAIN_CSV)


# In[ ]:

x = get_features(train)


# In[ ]:

y = get_target(train)


# In[ ]:

x_train, x_test, y_train, y_test = ttsplit(x,y)


# In[81]:

pipeline = make_pipeline(feature_selector,imputer,scaler, mlp)


# In[82]:

model = pipeline.fit(x_train, y_train)


# In[83]:

predictions = model.predict(x_test)


# In[84]:

mean_squared_error(predictions, y_test)


# In[85]:

mean_absolute_error(predictions, y_test)


# In[86]:

model.score(x_test, y_test)


# In[89]:

real_data = load_csv(TEST_CSV)


# In[90]:

real_features = get_features(real_data)


# In[91]:

real_predictions = model.predict(real_features)


# In[92]:

np.savetxt('/home/frug/neural_network_predictions.csv', real_predictions,delimiter=',')


# In[ ]:



