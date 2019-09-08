'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/8 23:18
@Author  :Zhangyunjia
@FileName: 5.3.4_precision_recall_curve.py
@Software: PyCharm
'''

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_curve
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=450, centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
