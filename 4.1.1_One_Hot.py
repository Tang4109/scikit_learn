import pandas as pd
import numpy as np
# np.set_printoptions(threshold=np.nan)
# pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
# pd.set_option('display.max_columns', None) # 可以填数字，填None表示'列'无限制
from IPython.display import display
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('introduction_to_ml_with_python-master/data/adult.data', header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                          'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
                          'native_country', 'income'])

data = data[['age', 'workclass', 'education', 'gender', 'hours_per_week',
             'native_country', 'occupation', 'income']]

# display(data.head())

# print(data.gender.value_counts())
# print('Original features:\n', list(data.columns))

data_dummies = pd.get_dummies(data)
# print('Features after get_dummies:\n',data_dummies.columns)
# print(data_dummies.head())
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values

print('X.shape:{}  y.shape{}'.format(X.shape, y.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)  # 生成规则
X_train_scaled = scaler.transform(X_train)  # 将规则应用于训练集
X_test_scaled = scaler.transform(X_test)  # 将规则应用于测试集

logreg = LogisticRegression(solver='lbfgs',max_iter=5000)
logreg.fit(X_train_scaled, y_train)
score = logreg.score(X_test_scaled, y_test)
print(score)
