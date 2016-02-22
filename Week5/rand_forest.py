# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import sklearn.cross_validation as cross_val
from sklearn.ensemble import RandomForestRegressor
from sklearn import grid_search
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

#Sex,Length,Diameter,Height,WholeWeight,ShuckedWeight,VisceraWeight,ShellWeight,Rings
dt = pd.read_csv('abalone.csv')
dt['Sex'] = dt['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

y = dt.Rings
del dt['Rings'] 
X = dt


grid = {'n_estimators': np.arange(1,51)}
cv = cross_val.KFold(y.size, n_folds=5, shuffle=True, random_state=1)
clf = RandomForestRegressor(random_state=1)
gs = grid_search.GridSearchCV(clf, grid, cv=cv, scoring = 'r2')
gs.fit(X, y)
for a in gs.grid_scores_:
     print(a.mean_validation_score,a.parameters)