from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# 导入数据
X, y = mglearn.datasets.make_wave(n_samples=100)
# 创造测试数据
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
# reshape(-1, 1)让line变成只有一列，行数自动计算
# endpoint如果是真，则一定包括stop，如果为False，一定不会有stop

# 分箱
bins = np.linspace(-3, 3, 11)
# 连续数据分箱离散化
which_bin = np.digitize(X, bins)
line_bin=np.digitize(line,bins)
# one-hot编码
encoder = OneHotEncoder(categories='auto', sparse=False)
# sparse若为True时，返回稀疏矩阵，否则返回数组
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
line_binned=encoder.transform(line_bin)
# 将原始特征加入新数据
X_combined = np.hstack([X, X_binned])
line_combined=np.hstack([line,line_binned])
#交互特征
X_product = np.hstack([X_binned, X*X_binned])
line_product=np.hstack([line_binned,line*line_binned])

#X_binned每一行的每个数乘以X的一行（唯一的一个数）


# 导入模型进行训练
reg = LinearRegression().fit(X_product, y)
plt.plot(line, reg.predict(line_product), label='linear regression product')
for bin in bins:
    plt.plot([bin,bin],[-3,3],':',c='k')


# reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
# plt.plot(line, reg.predict(line_binned), label='decision tree binned')
#
plt.plot(X[:, 0], y, 'o', c='k')
# plt.vlines(bins, -3, 3, linewidth=1, alpha=0.2)
plt.ylabel('regression output')
plt.xlabel('input feature')
plt.legend(loc='best')
plt.show()
