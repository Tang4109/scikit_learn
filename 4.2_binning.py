from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

X, y = mglearn.datasets.make_wave(n_samples=100)

line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)  # reshape(-1, 1)让line变成只有一列，行数自动计算
# 如果是真，则一定包括stop，如果为False，一定不会有stop
# print(line)
# reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
# plt.plot(line, reg.predict(line), label='decision tree')
# reg = LinearRegression().fit(X, y)
# plt.plot(line, reg.predict(line), label='linear regression')
#
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel('regression output')
# plt.xlabel('input feature')
# plt.legend(loc='best')
# plt.show()

bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins)

encoder = OneHotEncoder(categories='auto', sparse=False)  # sparse若为True时，返回稀疏矩阵，否则返回数组
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print(X_binned[:5])
print(X_binned.shape)

line_binned = encoder.transform(np.digitize(line, bins=bins))
print(line_binned.shape)

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned')

plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=0.2)
plt.ylabel('regression output')
plt.xlabel('input feature')
plt.legend(loc='best')
plt.show()
