import pandas as pd
import numpy as np
import math  as mh
import sklearn.metrics as metrics

dt = pd.read_csv('data-logistic.csv',header = None)
Y = dt[0].values
X1 = dt[1].values
X2 = dt[2].values

iter_num = 10000
w1 = w2 = 0
k = 0.1
ln = len(Y)

#regulirization
C=10.0 #C=10.0

i=0
dist = 100.0
for i in range(iter_num+1):
    delta1 = delta2 = l = 0
    while l < ln:
        delta1 += Y[l]*X1[l]*(1.0-1.0/(1.0+mh.exp(-Y[l]*(w1*X1[l]+w2*X2[l]))))
        delta2 += Y[l]*X2[l]*(1.0-1.0/(1.0+mh.exp(-Y[l]*(w1*X1[l]+w2*X2[l]))))
        l += 1
 #   delta1 = np.dot(Y,X1)*(1.0-1.0/(1.0+mh.exp(np.dot(-Y,w1*X1+w2*X2))))
 #   delta1 = np.dot(Y,X2)*(1.0-1.0/(1.0+mh.exp(np.dot(-Y,w1*X1+w2*X2))))
    w1n = w1 + k*delta1/ln - k*C*w1
    w2n = w2 + k*delta2/ln - k*C*w2
    dist = mh.sqrt((w1-w1n)**2 + (w2-w2n)**2)
    
    w1 = w1n
    w2 = w2n
    #print('i = %d dist = %.10f') % (i, dist)
    #print('w1 = %f w2 = %f') % (w1,w2)
    
    if (dist < 1e-5):
        break;

print('i = %d dist = %.10f') % (i, dist)
print('w1 = %f w2 = %f') % (w1,w2)

pred = []
l=0
while l < ln:
    pred.append(1./(1. + mh.exp(-w1*X1[l]-w2*X2[l])))
    l += 1
#pred = 1.0/(1.0 + mh.exp(np.dot(-w1,X1)+np.dot(-w2,X2)))
    
auc_score = metrics.roc_auc_score(Y,pred)
print ('AUC = %f') % auc_score