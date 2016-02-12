import pandas as pd
import numpy as np
import sklearn.cross_validation as cross_val
import sklearn.neighbors as neighbors
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston

boston = load_boston()

features = scale(boston.data)
target = boston.target
kf = cross_val.KFold(len(features),n_folds=5, shuffle=True, random_state=42)
res = []
for par in np.linspace(1.0, 10.0, num=200):
    print('p = %f') % par 
    estimator = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance',  p=par, metric='minkowski')
    score = cross_val.cross_val_score(estimator, features, target, cv = kf, scoring='mean_squared_error').mean()
    res.append(score) 
    print ('score = %f') % score
print(sorted(res))