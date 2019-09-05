'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/5 21:39
@Author  :Zhangyunjia
@FileName: 4.6_expert_knowledge.py
@Software: PyCharm
'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mglearn

# 加载数据
citibike = mglearn.datasets.load_citibike()

# 可视化
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
# 在处理时间序列数据时，这个函数的作用就是产生一个DatetimeIndex，就是时间序列数据的索引。
# freq='D',表示以自然日为单位
# plt.xticks(xticks, xticks.strftime('%a %m-%d'), rotation=90, ha='left')
# strftime() 函数接收以时间元组，并返回以可读字符串表示的当地时间
# ha='right'点在注释右边
# plt.plot(citibike, linewidth=1)
# plt.xlabel('Date')
# plt.ylabel('Rentals')
# plt.show()

# 提取目标值
y = citibike.values
X = citibike.index.astype("int64").values.reshape(-1, 1)
n_train = 184
def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    score = mean_squared_error(y_test, y_pred)
    print('均方误差：',score)
    plt.figure(figsize=(10,3))
    plt.xticks(range(0,len(X),8), xticks.strftime('%a %m-%d'), rotation=90, ha='left')
    plt.plot(range(n_train),y_train,c='k',label='train')
    plt.plot(range(n_train,len(y_test)+n_train),y_test,'-',label='test')
    plt.plot(range(n_train),y_pred_train,'--',label='prediction train')
    plt.plot(range(n_train,len(y_test)+n_train),y_pred,'--',label='prediction test')
    plt.legend(loc=(1.01,0))
    plt.xlabel('Date')
    plt.ylabel('Rentals')
    plt.show()

# regressor=RandomForestClassifier(n_estimators=100,random_state=0)
regressor=LinearRegression()
# eval_on_features(X,y,regressor)
X_hour=np.array(citibike.index.hour).reshape(-1,1)
X_hour_week=np.hstack([np.array(citibike.index.dayofweek).reshape(-1,1),X_hour])

eval_on_features(X_hour_week,y,regressor)






