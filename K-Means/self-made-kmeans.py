# coding = utf-8
"""
@author: zhou
@time:2019/10/9 15:35
@File: self-made-kmeans.py
"""

from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

k_means_data = pd.read_csv('k-meansdata.csv')
print(k_means_data.shape)
print(k_means_data.head())

data1 = k_means_data['V1'].values
data2 = k_means_data['V2'].values
X = np.array(list(zip(data1, data2)))
# 随机样本点
#  = np.random.random((200, 2))*10
plt.scatter(X[:, 0], X[:, 1], s=6)
print(X[:, 0])
print("+++++++++")
print(X[:, 1])
print("+++++++++++")
print(X)
# plt.show()


# 自制 kmeans 算法
def self_kmeans(data, k):
    m, n = data.shape
    results = np.empty(m)
    cores = np.copy(data[np.random.randint(0, m, size=k)])
    while True:

        for i in range(m):
            distance = np.linalg.norm(data[i] - cores, axis=1)
            result = np.argmin(distance)
            results[i] = result
        cores_old = deepcopy(cores)
        for i in range(k):
            points = [data[j] for j in range(m) if results[j] == i]
            cores[i] = np.mean(points, axis=0)

        if (cores_old == cores).all():
            return results, cores
