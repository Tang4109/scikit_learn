'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/8/19 15:53
@Author  :Zhangyunjia
@FileName: 3.4.1_pca_cancer.py
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
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_breast_cancer
import matplotlib.pyplot as plt
import mglearn
from sklearn.decomposition import PCA

# 处理训练数据
# 读入训练数据，sheetname=0表示第一个表单
cancer = load_breast_cancer()

# 对数据的训练集进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cancer.data)
# 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_scaled.shape)
print(X_pca.shape)

print(pca.components_)




# 画图
fig = plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc='best')
plt.gca().set_aspect('equal')  # get current axes
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()
