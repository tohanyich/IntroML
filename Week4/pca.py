# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

dt = pd.read_csv('close_prices.csv')
dt_test = pd.read_csv('djia_index.csv')
#Убирем столбец с датой
del dt['date']

pca = PCA(n_components = 10)
pca.fit(dt)

print(pca.explained_variance_ratio_ )
print(pca.explained_variance_ratio_[[0,1,2,3]].sum())

dt_new = pca.transform(dt)
first_component = dt_new[::,0]
print(pca.components_[0])
print(max(pca.components_[0]))

#pearson_corr
print(np.corrcoef(first_component,dt_test['^DJI']))