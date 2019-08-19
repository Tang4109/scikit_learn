'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/8/19 9:29
@Author  :Zhangyunjia
@FileName: 3.3.3_wrong_scaler.py
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn
# 处理训练数据
# 读入训练数据，sheetname=0表示第一个表单
X,y = make_blobs(n_samples=50,centers=5,random_state=4,cluster_std=2)
#先标准化再切分
ss = MinMaxScaler()
x_regular = ss.fit_transform(X)
X_trainx, X_testx, y_trainx, y_testx = train_test_split(x_regular, y, random_state=0)
# 将数据随机分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 对数据的训练集进行标准化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#单独对测试集进行缩放的错误做法
test_scaler=MinMaxScaler()
X_test_scaled_badly = test_scaler.fit_transform(X_test)

#先标准化再切分，与先切分再标准化之间有细微差别
print(X_test_scaled[:,0])
print(X_testx[:,0])
#画图
fig=plt.figure(figsize=(13,4))
ax0=fig.add_subplot(1,4,1)
ax0.scatter(X_train[:,0],X_train[:,1],c=mglearn.cm2(0),label='Training set',s=60)
ax0.scatter(X_test[:,0],X_test[:,1],marker='^',c=mglearn.cm2(1),label='Test set',s=60)
ax0.legend(loc='upper left')
ax0.set_title('Original Data')
ax0.set_xlabel('Feature 0')
ax0.set_ylabel('Feature 1')

ax1=fig.add_subplot(1,4,2)
ax1.scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0),label='Training set',s=60)
ax1.scatter(X_test_scaled[:,0],X_test_scaled[:,1],marker='^',c=mglearn.cm2(1),label='Test set',s=60)
ax1.legend(loc='upper left')
ax1.set_title('Scaled Data')
ax1.set_xlabel('Feature 0')
ax1.set_ylabel('Feature 1')

ax2=fig.add_subplot(1,4,3)
ax2.scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0),label='Training set',s=60)
ax2.scatter(X_test_scaled_badly[:,0],X_test_scaled_badly[:,1],marker='^',c=mglearn.cm2(1),label='Test set',s=60)
ax2.legend(loc='upper left')
ax2.set_title('Improperly Scaled Data')
ax2.set_xlabel('Feature 0')
ax2.set_ylabel('Feature 1')

ax3=fig.add_subplot(1,4,4)
ax3.scatter(X_trainx[:,0],X_trainx[:,1],c=mglearn.cm2(0),label='Training set',s=60)
ax3.scatter(X_testx[:,0],X_testx[:,1],marker='^',c=mglearn.cm2(1),label='Test set',s=60)
ax3.legend(loc='upper left')
ax3.set_title('pre Scaled Data')
ax3.set_xlabel('Feature 0')
ax3.set_ylabel('Feature 1')

plt.show()