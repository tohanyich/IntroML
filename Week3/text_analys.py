# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn.cross_validation as cross_val
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import grid_search

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
v = TfidfVectorizer()
X = v.fit_transform(newsgroups.data)
y = newsgroups.target

# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# cv = cross_val.KFold(y.size, n_folds=5, shuffle=True, random_state=241)
# clf = svm.SVC(kernel='linear', random_state=241)
# gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# gs.fit(X, y)
# 
# for a in gs.grid_scores_:
#     print(a.mean_validation_score,a.parameters)

clf = svm.SVC(C = 10.0, kernel='linear', random_state=241)
clf.fit(X, y)
print(pd.Series(clf.coef_.toarray().reshape(-1)).abs().nlargest(10).index)
#print(np.argsort(sorted(np.abs(clf.coef_.data)))[-10:])
index = pd.Series(clf.coef_.toarray().reshape(-1)).abs().nlargest(10).index
words = []
for i in index:
   words.append(v.get_feature_names()[i])
for w in sorted(words):
   print(w)