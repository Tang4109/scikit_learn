'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/8/17 8:54
@Author  :Zhangyunjia
@FileName: 3.3.2_data_transfer.py
@Software: PyCharm
'''
import pandas as pd  # 数据分析
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer

# 处理训练数据
# 读入训练数据，sheetname=0表示第一个表单
data_train = load_breast_cancer()
# 将数据随机分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_train.data, data_train.target, random_state=1)
# 数据变换
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.min(axis=0))
print(X_train_scaled.max(axis=0))
print(X_test_scaled.min(axis=0))
print(X_test_scaled.max(axis=0))

