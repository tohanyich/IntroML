# -*- coding: UTF-8 -*-

import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def sexToBinary(strSex):
    if strSex == 'male':
        return 1
    else:
        return 0

data = pandas.read_csv('titan.csv',index_col='PassengerId')
dataN = data
BinSex = dataN['Sex'].apply(lambda x: sexToBinary(str(x)))
dataN['BinSex'] = BinSex
#print(dataN)
del dataN[u'Name'],dataN[u'Sex'],dataN[u'SibSp'],dataN[u'Parch'],dataN[u'Ticket'],dataN[u'Cabin'],dataN[u'Embarked'] #,dataN[u'Survived'],

dataN = dataN.dropna()

target = dataN.Survived.values
#print(dataN)
del dataN[u'Survived']
features = dataN.values

clf = DecisionTreeClassifier(random_state=241)
lf = clf.fit(features, target)
importances = clf.feature_importances_
print(dataN.columns)
print(importances)