# coding = utf-8
"""
@author: zhou
@time:2019/9/17 19:28
@File: hr.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

df = pd.read_csv('HR.csv', index_col=None)
# 检测是否有缺失数据
df.isnull().any()
df.head()

# 数据分析
print(df.shape)

turnover_rate = df.left.value_counts() / len(df)
print(turnover_rate)

turnover_Summary = df.groupby('left')
print(turnover_Summary.mean())

# 相关性分析
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# 数据集字符串转换成数字
print(df.dtypes)

df["sales"] = df["sales"].astype('category')
df["salary"] = df["salary"].astype('category')

df["sales"] = df["sales"].cat.codes
df["salary"] = df["salary"].cat.codes

# 模型训练
target_name = 'left'
X = df.drop('left', axis=1)
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)

clf = DecisionTreeClassifier(
    criterion='entropy',
    #max_depth=3, # 定义树的深度, 可以用来防止过拟合
    min_weight_fraction_leaf=0.01 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
    )
clf = clf.fit(X_train, y_train)

clf_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
print("决策树 AUC = %2.2f" % clf_roc_auc)

# ROC 曲线
clf_fpr, clf_tpr, clf_thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

plt.figure()

# 决策树 ROC
plt.plot(clf_fpr, clf_tpr, label='Decision Tree (area = %0.2f)' % clf_roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

# 画出决策树的特征重要性
importances = clf.feature_importances_
feat_names = df.drop(['left'],axis=1).columns


indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by Decision Tree")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()