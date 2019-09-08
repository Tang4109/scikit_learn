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
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=4500, centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))

print(thresholds)
close_zero = np.argmin(np.abs(thresholds))  # 索引
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, label='threshold zero', fillstyle='none', c='k',
         mew=2)

plt.plot(precision, recall, label='precision_recall_curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()
