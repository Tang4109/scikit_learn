'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/8/20 17:52
@Author  :Zhangyunjia
@FileName: 3.5.1_kmeans_for_moons.py
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
from sklearn.datasets import make_blobs, load_breast_cancer, make_moons
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mglearn

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
#使用比数据维度更多的簇来对数据进行编码
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')  # cmap = colormap
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=60,
            marker='^',c=range(kmeans.n_clusters),linewidth=2,cmap='Paired')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()