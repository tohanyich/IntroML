import pandas as pd
import numpy as np
import sklearn.cross_validation as cross_val
from sklearn.svm import SVC

dt = pd.read_csv('svm-data.csv',header = None)
target = dt[0]
del dt[0]
feat = dt

clf = SVC(C = 10000, kernel='linear', random_state=241)
clf.fit(feat, target)

print(clf.support_)