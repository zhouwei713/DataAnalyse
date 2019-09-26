# coding = utf-8
"""
@author: zhou
@time:2019/9/26 19:23
@File: breast.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


breast = pd.read_csv('breast_data.csv')
breast.head()
print(breast.isnull().sum())
print(breast['diagnosis'].value_counts())

# 数据处理
breast.drop("id", axis=1, inplace=True)
breast['diagnosis']=breast['diagnosis'].map({'M': 1, 'B': 0})

# 数据拆分
breast_mean = list(breast.columns[1:11])
breast_se = list(breast.columns[11:21])
breast_worst = list(breast.columns[21:31])

# 相关性分析
breast_corr = breast[breast_mean].corr()
sns.heatmap(breast_corr, annot=True)
plt.show()

# 特征选择
breast_features = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']

# 拆分训练集和测试集
train, test = train_test_split(breast, test_size = 0.3)
# 抽取特征选择的数值作为训练和测试数据
X_train = train[breast_features]
y_train =train['diagnosis']
X_test = test[breast_features]
y_test =test['diagnosis']

# 归一化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 模型训练和预测
# 创建 SVM 分类器
model = SVC()
# 用训练集做训练
model.fit(X_train,y_train)
# 用测试集做预测
prediction=model.predict(X_test)
print('准确率: ', accuracy_score(prediction,y_test))

# 查看性能报告
print(classification_report(y_test, prediction))


