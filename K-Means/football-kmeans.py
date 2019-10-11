# coding = utf-8
"""
@author: zhou
@time:2019/10/11 11:42
@File: football-kmeans.py
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


team_data = pd.read_csv('football-team.csv')
print(team_data)

train_X = team_data[['2019年国际排名', '2019亚洲杯', '2015亚洲杯']]
print(train_X)

# 数据规范化
mm = preprocessing.MinMaxScaler()
train_x = mm.fit_transform(train_X)

# 手肘法
SS = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k).fit(train_x)
    SS.append(kmeans.inertia_)
plt.plot(range(2,10), SS)
plt.xlabel('K')
plt.ylabel('SS')
plt.show()

# 使用最佳的 k 值
kmeans = KMeans(n_clusters=4)

# kmeans 算法
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
# 合并聚类结果，插入到原数据中
result = pd.concat((team_data, pd.DataFrame(predict_y)), axis=1)
result.rename({0: u'聚类'}, axis=1, inplace=True)
print(result)

