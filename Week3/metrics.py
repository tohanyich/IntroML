import pandas as pd
import numpy as np
import sklearn.metrics as metrics

cls = pd.read_csv('classification.csv')

n=TP=FP=FN=TN = 0
while n < len(cls.true):
	if cls.pred[n] == 1:
		if cls.true[n] == 1:
			TP += 1
		else:
			FP += 1
	else:
		if cls.true[n] == 1:
			FN += 1
		else:
			TN += 1
	n += 1
print('TP=%d	FP=%d	FN=%d	TN=%d') % (TP,FP,FN,TN)

acc = metrics.accuracy_score(cls.true,cls.pred)
prec = metrics.precision_score(cls.true,cls.pred)
recall = metrics.recall_score(cls.true,cls.pred)
f_mes = metrics.f1_score(cls.true,cls.pred)

print('acc=%.2f	prec=%.2f	recall=%.2f	f_mes=%.2f') % (acc,prec,recall,f_mes)