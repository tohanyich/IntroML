# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import sklearn.cross_validation as cross_val
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

dt = pd.read_csv('gbm-data.csv')
y = dt.Activity.values

del dt['Activity']
X = dt.values
X_train,X_test,y_train,y_test = cross_val.train_test_split(X, y, test_size=0.8, random_state=241)

clf = RandomForestClassifier(n_estimators=36, random_state=241)
clf.fit(X_train, y_train)

test_loss = log_loss(y_test, clf.predict_proba(X_test)[:, 1])
    
print('test = %f') % test_loss