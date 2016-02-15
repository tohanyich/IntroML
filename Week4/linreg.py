# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

#FullDescription,LocationNormalized,ContractTime,SalaryNormalized
dt = pd.read_csv('salary-train.csv')
dt_test = pd.read_csv('salary-test-mini.csv')

#preprocessing
dt['FullDescription'] = dt['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))
dt['LocationNormalized'] = dt['LocationNormalized'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))
dt_test['FullDescription'] = dt_test['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))
dt_test['LocationNormalized'] = dt_test['LocationNormalized'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))

Tfid = TfidfVectorizer(min_df = 5)
X_Tfid= Tfid.fit_transform(dt['FullDescription'])
X_Tfid_test= Tfid.transform(dt_test['FullDescription'])

dt['LocationNormalized'].fillna('nan', inplace=True)
dt['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
X_train = enc.fit_transform(dt[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(dt_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X = hstack([X_Tfid,X_train])
y = dt['SalaryNormalized']
#Learning
clf = Ridge(alpha=1.0)
clf.fit(X, y)

X_test = hstack([X_Tfid_test,X_test_categ])
P = clf.predict(X_test)
print(P)
