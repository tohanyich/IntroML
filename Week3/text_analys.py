import numpy as np
import sklearn.cross_validation as cross_val
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import grid_search

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = cross_val.KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)