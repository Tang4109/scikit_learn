'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    :2019/9/4 9:32
@Author  :Zhangyunjia
@FileName: 4.4_singleVariable_nonlinear_transition.py
@Software: PyCharm
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

#引入数据
rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

#exp变换
X = rnd.poisson(10 * np.exp(X_org))  # 产生泊松分布随机数
y = np.dot(X_org, w)
# print(X[:,0][:10])
# print(np.bincount(X[:,0]))

#直方图
# bins=np.bincount(X[:,0])
# plt.bar(range(len(bins)),bins,color='b')
# plt.ylabel('number of appearances')
# plt.xlabel('value')
# plt.show()

#训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#log变换
X_log=np.log(X+1)
X_train_log=np.log(X_train+1)
X_test_log=np.log(X_test+1)
score=Ridge().fit(X_train,y_train).score(X_test,y_test)
score_log=Ridge().fit(X_train_log,y_train).score(X_test_log,y_test)

print('Test score:{:.3f}'.format(score))
print('Test score:{:.3f}'.format(score_log))

plt.hist(X_train_log[:,0],bins=25,color='gray')
plt.ylabel('number of appearances')
plt.xlabel('value')
plt.show()
