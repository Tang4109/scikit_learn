import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.datasets import make_circles
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
y_named = np.array(['blue', 'red'])[y]
X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)
pred = gbrt.predict(X_test)
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
print(X_test.shape)
print(gbrt.decision_function(X_test) > 0)
print(pred)
print(np.all(gbrt.classes_[greater_zero] == pred))

decision_function = gbrt.decision_function(X_test)
print(np.min(decision_function))
print(np.max(decision_function))
# 画图
fig, axes = plt.subplots(1, 2, figsize=(13, 5))  # 创建一张画布并且创建子图

mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=0.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=0.4, cm=mglearn.ReBl)
for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
cbar = plt.colorbar(scores_image, ax=axes.tolist())  # 颜色棒，颜色越深代表置信度越高
axes[0].legend(['Test class 0','Test class 1','Train class 0','Train class 1'],ncol=4,loc=(0.1,1.1))
plt.show()
