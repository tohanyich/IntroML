import pandas as pd
import numpy as np
import sklearn.cross_validation as cross_val
import sklearn.neighbors as neighbors
from sklearn.preprocessing import scale

data = pd.DataFrame(data = pd.read_csv('wine_data.txt'))
target = data.Class.values
del data['Class']
#features = data.values
features = scale(data.values)
#print('t = %d, f = %d') % (len(target),len(features))
kfold = cross_val.KFold(len(features),n_folds=5, shuffle=True, random_state=42)
k=1
res = []
while k<=50:
    print('k = %d') % k
    estimator = neighbors.KNeighborsClassifier(n_neighbors=k)
    score = cross_val.cross_val_score(estimator, features, target, cv = kfold).mean()
    res.append(score)
    print("Score with the entire dataset = %.10f" % score)
    k+=1
print(sorted(res))