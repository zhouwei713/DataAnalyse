# coding = utf-8
"""
@author: zhou
@time:2019/9/11 14:06
@File: second_hand_cars.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# 导入数据
df = pd.read_csv('knn-regression.csv')
print(df)

# 数据处理，独热编码
df_new = pd.get_dummies(df, columns=['Type'])
df_new = pd.get_dummies(df_new, columns=['Color'])
print(df_new)

# 查看每列数值信息
for col in df.columns:
    print(df[col].value_counts())

# 删除无效列
df_new.drop(['Brand'], axis=1, inplace=True)

# 数据关联性分析
matrix = df_new.corr()
plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, square=True)
plt.show()

# 训练及预测
# 选择特征和预测值
X = df_new[['Construction Year', 'Days Until MOT', 'Color_grey']]
y = df_new['Ask Price'].values.reshape(-1, 1)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

# 数据规范化
ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train_ss, y_train)
y_pred = knn.predict(X_test_ss)

# 可视化分析预测情况
plt.scatter(y_pred, y_test)
plt.xlabel("Prediction")
plt.ylabel("Real Value")
diag = np.linspace(500, 1200, 100)
plt.plot(diag, diag, '-r')
plt.show()



