from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures

# 导入数据
X, y = mglearn.datasets.make_wave(n_samples=100)
poly=PolynomialFeatures(degree=10,include_bias=False)
poly.fit(X)
X_poly=poly.transform(X)

# 创造测试数据
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
# reshape(-1, 1)让line变成只有一列，行数自动计算
# endpoint如果是真，则一定包括stop，如果为False，一定不会有stop
line_poly=poly.transform(line)

# # 导入模型进行训练
reg = LinearRegression().fit(X_poly, y)
plt.plot(line, reg.predict(line_poly), label='Polynomial linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('regression output')
plt.xlabel('input feature')
plt.legend(loc='best')
plt.show()

