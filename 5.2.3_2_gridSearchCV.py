'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/7 9:24
@Author  :Zhangyunjia
@FileName: 5.2.3_2_gridSearchCV.py
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

param_grid={'C':[0.001, 0.01, 0.1, 1, 10, 100],'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
grid_search=GridSearchCV(SVC(),param_grid,cv=5)

grid_search.fit(X_train,y_train)
score=grid_search.score(X_test,y_test)
print(score)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)

results=pd.DataFrame(grid_search.cv_results_)
print(results.head())

scores=np.array(results.mean_test_score).reshape(6,6)
mglearn.tools.heatmap(scores,xlabel='gamma',xticklabels=param_grid['gamma'],ylabel='C',yticklabels=param_grid['C'],cmap='viridis')
plt.show()


fig,axes=plt.subplots(1,3,figsize=(13,5))
param_grid_linear={'C':np.linspace(1,2,6),'gamma':np.linspace(1,2,6)}
param_grid_one_log={'C':np.linspace(1,2,6),'gamma':np.logspace(-3,2,6)}
param_grid_range={'C':np.logspace(-3,2,6),'gamma':np.logspace(-3,2,6)}
for param_grid,ax in zip([param_grid_linear,param_grid_one_log,param_grid_range],axes):
    grid_search=GridSearchCV(SVC(),param_grid,cv=5)
    grid_search.fit(X_train,y_train)
    scores=grid_search.cv_results_['mean_test_score'].reshape(6,6)
    #绘图
    scores_image=mglearn.tools.heatmap(scores,xlabel='gamma',ylabel='C',xticklabels=param_grid['gamma'],yticklabels=param_grid['C'],cmap='viridis',ax=ax)

plt.colorbar(scores_image,ax=axes.tolist())
plt.show()


