# coding = utf-8
"""
@author: zhou
@time:2019/9/23 10:51
@File: text_clf.py
"""

import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import metrics
import jieba
from sklearn.metrics import confusion_matrix


def cut_word(file):
    text = open(file, 'r', encoding='gb18030').read()
    stopword = [line.strip() for line in open(r'text/stop/stopword.txt', encoding='utf-8').readlines()]
    text_segd = jieba.cut(text.strip())
    seg_word = ''
    for word in text_segd:
        if word not in stopword:
            seg_word += word + ' '
    return seg_word


def load_file(file_path, label):
    file_list = os.listdir(file_path)
    words_list, labels_list = [], []
    for f in file_list:
        file = file_path + '/' + f
        words_list.append(cut_word(file))
        labels_list.append(label)
    return words_list, labels_list


# 训练数据
train_words_list1, train_labels1 = load_file('text/train/女性', '女性')
train_words_list2, train_labels2 = load_file('text/train/体育', '体育')
train_words_list3, train_labels3 = load_file('text/train/文学', '文学')
train_words_list4, train_labels4 = load_file('text/train/校园', '校园')

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

# 测试数据
test_words_list1, test_labels1 = load_file('text/test/女性', '女性')
test_words_list2, test_labels2 = load_file('text/test/体育', '体育')
test_words_list3, test_labels3 = load_file('text/test/文学', '文学')
test_words_list4, test_labels4 = load_file('text/test/校园', '校园')

test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

# 文档转向量
tf = TfidfVectorizer()

train_features = tf.fit_transform(train_words_list)
test_features = tf.transform(test_words_list)

clf = MultinomialNB(alpha=0.001)
clf.fit(train_features, train_labels)
predicted_labels = clf.predict(test_features)

# 计算准确率
print('准确率为：', metrics.accuracy_score(test_labels, predicted_labels))

# 混淆矩阵
print(confusion_matrix(test_labels, predicted_labels, labels=['女性', '体育', '文学', '校园']))

