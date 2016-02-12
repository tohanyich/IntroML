import pandas as pd
import numpy as np
import sklearn.cross_validation as cross_val
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train = pd.read_csv('perc_train.csv',header = None)
test  = pd.read_csv('perc_test.csv',header = None)

target_tr = train[0]
target_ts = test[0]
del train[0], test[0]
feat_tr = train
feat_ts = test

scaler = StandardScaler()
feat_tr_scaled = scaler.fit_transform(feat_tr)
feat_ts_scaled = scaler.fit_transform(feat_ts)

clf = Perceptron(random_state = 241)

clf.fit(feat_tr,target_tr)
predict_before = clf.predict(feat_ts)
score_before = accuracy_score(target_ts, predict_before)
clf.fit(feat_tr_scaled,target_tr)
predict_after = clf.predict(feat_ts_scaled)
score_after = accuracy_score(target_ts, predict_after)

print('before %f, after %f, res %f ') % (score_before,score_after, score_after - score_before)