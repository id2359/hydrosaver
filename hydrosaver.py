import pandas as pd
from sklearn import linear_model

def convert_to_float(df):
    return df.apply(pd.to_numeric, errors='coerce')

TARGET_COLUMN = 'target'
COLUMNS_TO_EXCLUDE = ['timestamp',
                      'DIC88023.PV',
                      'II88151.PV',
                      'II88152.PV',
                      'FV88156.PV',
                      'FV88043.PV',
                      'FV88044.PV']


lm = linear_model.LinearRegression()

train = pd.read_csv("train.csv", header=0)

x = convert_to_float(train[train.columns.difference([TARGET_COLUMN]+COLUMNS_TO_EXCLUDE)].copy())

print "x columns = %s" % x.columns


y = convert_to_float(train[[TARGET_COLUMN]])

print "y columns = %s" % y.columns


model = lm.fit(x, y)










