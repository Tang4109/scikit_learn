'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/8/19 9:29
@Author  :Zhangyunjia
@FileName: 3.3.4_scaler_svc.py
@Software: PyCharm
'''
import pandas as pd  # 数据分析
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs,load_breast_cancer
import matplotlib.pyplot as plt
import mglearn
# 处理训练数据
# 读入训练数据，sheetname=0表示第一个表单
data_train =load_breast_cancer()
# 将数据随机分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_train.data, data_train.target, random_state=0)
# 对数据的训练集进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 训练
model = SVC(C=100)
clt = model.fit(X_train_scaled, y_train)

# 预测
y_pre = clt.predict(X_test_scaled)
# 得分
score_train = clt.score(X_train_scaled, y_train)  # 训练集得分
score_test = clt.score(X_test_scaled, y_test)  # 测试集得分
# 结果显示
print('训练数与预测数据的准确率')
print(score_train, score_test)
