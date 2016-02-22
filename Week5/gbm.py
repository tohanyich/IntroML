# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import sklearn.cross_validation as cross_val
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
#%matplotlib inline
def heldout_score(clf, X_test, y_test):
    """compute deviance scores on ``X_test`` and ``y_test``. """
    score = np.zeros((250,), dtype=np.float64)
    for i, y_pred in enumerate(clf.predict_proba(X_test)):
        s_test = 1/(1 + np.exp(-y_pred))
        score[i] = log_loss(y_test, s_test)
    return score

dt = pd.read_csv('gbm-data.csv')
y = dt.Activity.values

del dt['Activity']
X = dt.values
X_train,X_test,y_train,y_test = cross_val.train_test_split(X, y, test_size=0.8, random_state=241)

for lr in [0.2]:#[1, 0.5, 0.3, 0.2, 0.1]:
   
    clf = GradientBoostingClassifier(learning_rate = lr, n_estimators=250, verbose=True, random_state=241)
    clf.fit(X_train, y_train)
    
    train_loss = heldout_score(clf, X_train, y_train)
    test_loss = heldout_score(clf, X_test, y_test)
    print(sorted(test_loss))
    print(test_loss.tolist().index(min(test_loss)))
    
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()