# coding = utf-8
"""
@author: zhou
@time:2019/9/26 19:26
@File: mushrooms.py
"""

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 导入数据
mush = pd.read_csv('mushrooms.csv')
mush.head()

# 特征编码
mush_encoded = pd.get_dummies(mush)
print(mush_encoded.head())

# 提取特征和标签
X_mush = mush_encoded.iloc[:, 2:]
y_mush = mush_encoded.iloc[:, 1]


# 主成分分析降维
pca = PCA(n_components=10, whiten=True, random_state=42)
svc = SVC(kernel='linear', class_weight='balanced')
model = make_pipeline(pca, svc)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_mush, y_mush,
                                                random_state=42)

# 查看 param_grid 的 key 值
print(model.get_params().keys())

# 通过网络搜索寻找最优参数（惩罚系数 C）
param_grid = {'svc__C': [1, 5, 10, 50]}
grid = GridSearchCV(model, param_grid)
grid.fit(X_train, y_train)
print(grid.best_params_)

# 获取最优模型
svm_model = grid.best_estimator_
yfit = svm_model.predict(X_test)

# 性能报告
print(classification_report(y_test, yfit))

# 混淆矩阵
mat = confusion_matrix(y_test, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')



