# coding = utf-8
"""
@author: zhou
@time:2019/9/10 17:05
@File: movie.py
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("movie_dataset.txt")
print(df.info())  # 查看整体数据
print(df.head())


#  转换函数
def trans(x):
    if x == '动作':
        x = 0
    else:
        x = 1
    return x


df['movie_types'] = df['movie_types'].apply(trans)

# df1['movie_types'].apply(lambda x: 0 if x == '动作' else 1)  # 更快捷的转换方法

# 数据向量化
feature = df[['fighting_lens', 'kissing_lens']].values
label = df['movie_types'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(feature, label, random_state=2002)

# 训练模型并预测
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predict_y = knn.predict(X_test)
print("KNN 准确率", accuracy_score(y_test, predict_y))

# 画散点图
sns.scatterplot(x='fighting_lens', y='kissing_lens', hue='movie_types', data=df)
plt.show()

# 交叉验证
X_train_new = X_train[:18]
X_train_validation = X_train[18:]

for k in range(1, 15, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predict_y = knn.predict(X_test)
    print("K为%s的准确率" % k, accuracy_score(y_test, predict_y))
