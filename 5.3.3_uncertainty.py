'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/8 22:42
@Author  :Zhangyunjia
@FileName: 5.3.3_uncertainty.py
@Software: PyCharm
'''

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import mglearn
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=450, centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_train)

# mglearn.plots.plot_decision_threshold()
# plt.show()

print(classification_report(y_test, svc.predict(X_test)))

y_pred_lower_threshold=svc.decision_function(X_test)>-0.8
print(classification_report(y_test,y_pred_lower_threshold))
