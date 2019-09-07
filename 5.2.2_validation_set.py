'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/7 8:29
@Author  :Zhangyunjia
@FileName: 5.2.2_validation_set.py
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
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import mglearn


iris=load_iris()
X=iris.data
y=iris.target
# 将数据划分为训练+验证集、测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)
# 将训练+验证集划分为训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=0)
print('数据集的大小：')
print(X_train.shape[0],X_valid.shape[0],X_test.shape[0])

best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for c in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(C=c, gamma=gamma)
        svm.fit(X_train,y_train)
        score=svm.score(X_valid,y_valid)
        if score > best_score:
            best_score = score
            best_parameters = {'C': c, 'gamma': gamma}



#在训练集和验证集上重新构建模型
svm=SVC(**best_parameters)
svm.fit(X_trainval,y_trainval)
test_score=svm.score(X_test,y_test)
print('Best score:', best_score)
print('Best parameters:', best_parameters)
print('Test score:', test_score)



