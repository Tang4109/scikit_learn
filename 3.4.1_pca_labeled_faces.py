'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/8/19 16:49
@Author  :Zhangyunjia
@FileName: 3.4.1_pca_labeled_faces.py
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
from sklearn.datasets import make_blobs, load_breast_cancer, fetch_lfw_people
import matplotlib.pyplot as plt
import mglearn
from sklearn.decomposition import PCA

# mglearn.plots.plot_pca_whitening()
# plt.show()

# 处理训练数据
# 读入训练数据
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
# print(people.images.shape)
# print(len(people.target_names))
# print(people.target_names)
# print(people.target)

counts = np.bincount(people.target)
# print(counts)
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count, end='  '))
    # if ((i + 1) % 3 == 0):  # 每两个换行一次
    #     print()

mask = np.zeros(people.target.shape, dtype=np.bool)
# 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素返回一个新的无元素重复的元组或者列表
# print(np.unique(people.target))
for target in np.unique(people.target):
    # print(np.where(people.target == target))
    # print(np.where(people.target == target)[0])
    # print(np.where(people.target == target)[0][:50])
    mask[np.where(people.target == target)[0][:50]] = 1  # 每人取50张图像
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255  # 将灰度值缩放到0-1之间
# 将数据随机分为训练集和测试集,stratify使类的分布更平衡
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
pca = PCA(n_components=100, whiten=True, random_state=0)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
score = knn.score(X_test_pca, y_test)
print(score)
print()
# 画图
image_shape = people.images[0].shape
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for target,image,ax in zip(people.target,people.images,axes.ravel()): # zip打包为元组
    ax.imshow(image)
    ax.set_title(people.target_names[target])

# 人脸数据集前15个主成分的成分向量
print(pca.components_.shape)
fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(87, 65), cmap='viridis')
    ax.set_title('{}.component'.format(i + 1))

mglearn.plots.plot_pca_faces(X_train, X_test, image_shape=(87, 65))
plt.figure()
mglearn.discrete_scatter(X_train_pca[:,0],X_train_pca[:,1],y_train)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()
