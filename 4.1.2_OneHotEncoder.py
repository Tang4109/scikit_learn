import pandas as pd
pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
pd.set_option('display.max_columns', None) # 可以填数字，填None表示'列'无限制
from IPython.display import display

demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                        'Categorical Feature': ['socks', 'fox', 'socks', 'box']})

display(demo_df)

dummies_demo_df = pd.get_dummies(demo_df,columns=['Integer Feature','Categorical Feature'])
print(dummies_demo_df)