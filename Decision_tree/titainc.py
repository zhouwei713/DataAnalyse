# coding = utf-8
"""
@author: zhou
@time:2019/9/17 16:20
@File: titainc.py
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus


df = pd.read_csv('titanic_data.csv')
data = df.copy()
data['age'].fillna(df['age'].median(skipna=True), inplace=True)
data.drop(columns=['cabin'], inplace=True)
data['embarked'].fillna(df['embarked'].value_counts().idxmax(), inplace=True)
data.dropna(axis=0, how='any', inplace=True)
data.isnull().sum()  # 查看是否还有空值
data['alone']=np.where((data["sibsp"]+data["parch"])>0, 0, 1)
data.drop('sibsp', axis=1, inplace=True)
data.drop('parch', axis=1, inplace=True)
data =pd.get_dummies(data, columns=["embarked","sex"])
data.drop('name', axis=1, inplace=True)
data.drop('ticket', axis=1, inplace=True)

# 特征选择
feature = ["age","fare","alone","pclass","embarked_C","embarked_S", "embarked_Q","sex_male", "sex_female"]

X = data[feature]
y = data['survived']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# 构建决策树
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# 预测
pred_labels = clf.predict(X_test)

# 查看准确率
print("Accuracy:",accuracy_score(y_test, pred_labels))

# 决策树可视化

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')



