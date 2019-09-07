'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/7 7:55
@Author  :Zhangyunjia
@FileName: 5.2.1_simple_grid_search.py
@Software: PyCharm
'''

import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import mglearn

iris = load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for c in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(C=c, gamma=gamma)
        svm.fit(X_train,y_train)
        score=svm.score(X_test,y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'C': c, 'gamma': gamma}

print('Best score:', best_score)
print('Best parameters:', best_parameters)
