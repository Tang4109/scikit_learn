from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)
# decision_function
print(gbrt.decision_function(X_test).shape)
print(gbrt.decision_function(X_test)[:6, :])
# predict_proba
print(gbrt.predict_proba(X_test)[:6, :])

# 通过计算argmax来再现预测结果
print(np.argmax(gbrt.decision_function(X_test), axis=1))
print(np.argmax(gbrt.predict_proba(X_test), axis=1))
print(gbrt.predict(X_test))
#逻辑回归做分类
logreg=LogisticRegression()
named_target=iris.target_names[y_train]
logreg.fit(X_train,named_target)
print(logreg.classes_)
print(logreg.predict(X_test)[:10])
argmax_dec_func=np.argmax(logreg.decision_function(X_test),axis=1)
print(argmax_dec_func[:10])
print(logreg.classes_[argmax_dec_func][:10])
print(np.all(logreg.classes_[argmax_dec_func][:10]==logreg.predict(X_test)[:10]))




