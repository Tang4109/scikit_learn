'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/4 23:36
@Author  :Zhangyunjia
@FileName: 4.5.3_RFE.py
@Software: PyCharm
'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
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


#训练模型
select=RFE(RandomForestClassifier(n_estimators=100,random_state=42),n_features_to_select=40)
select.fit(X_train,y_train)
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
score=LogisticRegression().fit(X_train_rfe,y_train).score(X_test_rfe,y_test)
print(score)




# 可视化
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('sample index')
plt.show()