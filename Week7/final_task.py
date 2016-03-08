# -*- coding: UTF-8 -*-
import pandas as pd
import sklearn.cross_validation as cross_val

features = pd.read_csv('./features.csv', index_col='match_id')
target = features[u'radiant_win']

del features[u'duration'],features[u'radiant_win'], features[u'tower_status_radiant']
del features[u'tower_status_dire'],features[u'barracks_status_radiant'], features[u'barracks_status_dire']

str_num,col_num = features.shape
col_names = features.columns

#Выведем столбцы с пропусками и обработаем их
for name in col_names:
    num = features[name].count()
    if num < str_num:
        print('%d Null elements in column %s') %(str_num - num, name)
        features[name].fillna(value = 0)

#Обучение градиентным бустингом
kfold = cross_val.KFold(len(features),n_folds=5, shuffle=True