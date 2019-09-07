'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/7 15:34
@Author  :Zhangyunjia
@FileName: 5.2.3_4_nested_cross_validation.py
@Software: PyCharm
'''

import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut, \
    ShuffleSplit, StratifiedShuffleSplit, GroupKFold, GridSearchCV,ParameterGrid,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import mglearn

iris = load_iris()
X = iris.data
y = iris.target
param_grid = [
    {
        'kernel': ['rbf'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        'kernel': ['linear'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
]
# scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), X, y, cv=5)
# print(scores)
# print(scores.mean())


def nested_cv(X,y,inner_cv,outer_cv,Classifier,parameter_grid):
    outer_scores=[]
    for training_samples,test_samples in outer_cv.split(X,y):
        best_parms={}
        best_score=-np.inf
        for parameters in parameter_grid:
            cv_scores=[]
            for inner_train,inner_test in inner_cv.split(X[training_samples],y[training_samples]):
                clf=Classifier(**parameters)
                clf.fit(X[inner_train],y[inner_train])
                score=clf.score(X[inner_test],y[inner_test])
                cv_scores.append(score)
            mean_score=np.mean(cv_scores)
            if mean_score>best_score:
                best_score=mean_score
                best_parms=parameters
        clf=Classifier(**best_parms)
        clf.fit(X[training_samples],y[training_samples])
        outer_scores.append(clf.score(X[test_samples],y[test_samples]))
    return np.array(outer_scores)


# print(param_grid)
# print(ParameterGrid(param_grid))
# for parameters in ParameterGrid(param_grid):
#     print(parameters)


scores = nested_cv(X,y,StratifiedKFold(5),StratifiedKFold(5),SVC,ParameterGrid(param_grid))
print(scores)