'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/6 8:49
@Author  :Zhangyunjia
@FileName: 5.1_cross_validation.py
@Software: PyCharm
'''
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut, \
    ShuffleSplit, StratifiedShuffleSplit, GroupKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mglearn

iris = load_iris()
logreg = LogisticRegression()

# 分层K折交叉验证
scores = cross_val_score(logreg, iris.data, iris.target, cv=3)
print('分层K折交叉验证得分：', scores)
print('分层K折交叉验证平均得分：', scores.mean())

# K折交叉验证
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print('K折交叉验证得分：', cross_val_score(logreg, iris.data, iris.target, cv=kfold))

# 留一法交叉验证
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print('留一法交叉验证迭代次数：', len(scores))
print('留一法交叉验证平均得分：', scores.mean())

# 打乱划分交叉验证
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print('打乱划分交叉验证得分：', scores)

# 分层打乱划分交叉验证
shuffle_split = StratifiedShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print('分层打乱划分交叉验证得分：', scores)

# 分组交叉验证
X, y = make_blobs(n_samples=12, random_state=0)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print('分组交叉验证得分：',scores)

