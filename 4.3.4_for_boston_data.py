from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


boston=load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)


#数据缩放
scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#多项式特征
poly=PolynomialFeatures(degree=2).fit(X_train_scaled)#degree=2表示次数最高是2
X_train_poly=poly.transform(X_train_scaled)
X_test_poly=poly.transform(X_test_scaled)

# print(X_train.shape)
# print(X_train_poly.shape)
# print(poly.get_feature_names())

#训练Ridge()
ridge=Ridge()
ridge.fit(X_train_scaled,y_train)
print(ridge.score(X_test_scaled,y_test))
ridge.fit(X_train_poly,y_train)
print(ridge.score(X_test_poly,y_test))

#训练RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100)
rf.fit(X_train_scaled,y_train)
print(rf.score(X_test_scaled,y_test))
rf.fit(X_train_poly,y_train)
print(rf.score(X_test_poly,y_test))







