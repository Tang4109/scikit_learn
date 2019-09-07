'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/7 10:34
@Author  :Zhangyunjia
@FileName: 5.2.3_3_gridSearch_within_innerKernel.py
@Software: PyCharm
'''

import warnings


warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut, \
    ShuffleSplit, StratifiedShuffleSplit, GroupKFold,GridSearchCV
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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

param_grid=[
    {
    'kernel':['rbf'],
    'C':[0.001, 0.01, 0.1, 1, 10, 100],
    'gamma':[0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        'kernel': ['linear'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
]
grid_search=GridSearchCV(SVC(),param_grid,cv=5)
grid_search.fit(X_train,y_train)
score=grid_search.score(X_test,y_test)

print(score)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)

results=pd.DataFrame(grid_search.cv_results_)
print(results.head())








