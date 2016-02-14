import pandas as pd
import numpy as np
import sklearn.metrics as metrics

def get_res_array(perc_arr, rec_arr):
    res_arr = []
    n = 0
    while n < len(perc_arr):
        if rec_arr[n] >= 0.7:
            res_arr.append(perc_arr[n])
            #print('percision = %f recall =%f') % (perc_arr[n], rec_arr[n])
        n += 1
    return sorted(res_arr,reverse = True)

cls = pd.read_csv('scores.csv')

AUC_log = float(metrics.roc_auc_score(cls.true, cls.score_logreg))
AUC_svm = metrics.roc_auc_score(cls.true, cls.score_svm)
AUC_knn = metrics.roc_auc_score(cls.true, cls.score_knn)
AUC_tree = metrics.roc_auc_score(cls.true, cls.score_tree)

print('log = %.2f svm = %.2f knn = %.2f tree = %.2f') % (AUC_log, AUC_svm, AUC_knn, AUC_tree)

perc_log, rec_log, tres_log = metrics.precision_recall_curve(cls.true, cls.score_logreg)
perc_svm, rec_svm, tres_svm = metrics.precision_recall_curve(cls.true, cls.score_svm)
perc_knn, rec_knn, tres_knn = metrics.precision_recall_curve(cls.true, cls.score_knn)
perc_tree, rec_tree, tres_tree = metrics.precision_recall_curve(cls.true, cls.score_tree)

res_log = get_res_array(perc_log, rec_log)
res_svm = get_res_array(perc_svm, rec_svm)
res_knn = get_res_array(perc_knn, rec_knn)
res_tree = get_res_array(perc_tree, rec_tree)

print('log = %.2f svm = %.2f knn = %.2f tree = %.2f') % (res_log[0], res_svm[0], res_knn[0], res_tree[0])