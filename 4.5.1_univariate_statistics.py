'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/4 10:24
@Author  :Zhangyunjia
@FileName: 4.5.1_univariate_statistics.py
@Software: PyCharm
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression

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

# 特征选择
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
X_test_selected=select.transform(X_test)
#训练模型
lr=LogisticRegression()
lr.fit(X_train,y_train)
score=lr.score(X_test,y_test)
print(score)
lr.fit(X_train_selected,y_train)
score_selected = lr.score(X_test_selected,y_test)
print(score_selected)




# mask = select.get_support()
# print(mask)
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')  # reshape(1,-1)只有一行，列数自动计算
# plt.xlabel('Sample index')
# plt.show()
