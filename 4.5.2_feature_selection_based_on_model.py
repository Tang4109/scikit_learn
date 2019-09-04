'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/4 22:31
@Author  :Zhangyunjia
@FileName: 4.5.2_feature_selection_based_on_model.py
@Software: PyCharm
'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# 加载数据
cancer = load_breast_cancer()

# 获得确定性的随机数
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

# 添加噪声
X_w_noise = np.hstack([cancer.data, noise])
X = X_w_noise
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 模型训练
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
# print(X_train.shape)
# print(X_train_l1.shape)
X_test_l1=select.transform(X_test)
score=LogisticRegression().fit(X_train_l1,y_train).score(X_test_l1,y_test)
print(score)


# 可视化
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('sample index')
plt.show()


