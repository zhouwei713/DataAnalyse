# coding = utf-8
"""
@author: zhou
@time:2019/9/10 18:36
@File: digits.py
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 加载数据
digits = datasets.load_digits()
data = digits.data
# 查看整体数据
print(data.shape)
# 查看第一幅图像
print(digits.images[0])
# 第一幅图像代表的数字含义
print(digits.target[0])
# 将第一幅图像显示出来
plt.gray()
plt.imshow(digits.images[0])
plt.show()

#  数值规范化
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, random_state=2002)
# 采用 Z-Score 规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

# 创建分类器并预测
knn = KNeighborsClassifier()
knn.fit(train_ss_x, train_y)
predict_y = knn.predict(test_ss_x)
print("KNN 准确率: %.4lf" % accuracy_score(test_y, predict_y))




