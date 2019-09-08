'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/8 11:31
@Author  :Zhangyunjia
@FileName: 5.3.2_confusion_matrix.py
@Software: PyCharm
'''

import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs, load_digits
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut, \
    ShuffleSplit, StratifiedShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


digits = load_digits()
y = digits.target == 9
X = digits.data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)  # Dummy 分类器完全忽略输入数据
# most_frequent: 预测值是出现频率最高的类别
pred_most_frequent = dummy_majority.predict(X_test)
# print(pred_most_frequent)
print(np.unique(pred_most_frequent))
print(dummy_majority.score(X_test, y_test))

tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print(tree.score(X_test, y_test))

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print(dummy.score(X_test, y_test))

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print('logreg score',logreg.score(X_test, y_test))

print('dummy_majority')
confusion=confusion_matrix(y_test,pred_most_frequent)
print(confusion)

print('DecisionTreeClassifier')
confusion=confusion_matrix(y_test,pred_tree)
print(confusion)

print('DummyClassifier')
confusion=confusion_matrix(y_test,pred_dummy)
print(confusion)

print('LogisticRegression')
confusion=confusion_matrix(y_test,pred_logreg)
print(confusion)

print('--------------------------------------------')
from sklearn.metrics import f1_score
print('f1 score pred_most_frequent:',f1_score(y_test,pred_most_frequent))
print('f1 score pred_dummy:',f1_score(y_test,pred_dummy))
print('f1 score pred_tree:',f1_score(y_test,pred_tree))
print('f1 score pred_logreg:',f1_score(y_test,pred_logreg))

print('--------------------------------------------')
from sklearn.metrics import classification_report
print(classification_report(y_test,pred_most_frequent,target_names=['not nine','nine']))
print(classification_report(y_test,pred_dummy,target_names=['not nine','nine']))
print(classification_report(y_test,pred_logreg,target_names=['not nine','nine']))


















