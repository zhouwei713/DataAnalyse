# coding = utf-8
"""
@author: zhou
@time:2019/9/21 13:47
@File: sentiment.py
"""

import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

kk = GaussianNB().partial_fit()


# 导入数据函数
def load_data(file, is_positive=None):
    reviews, labels = [], []
    with open(file, 'r', encoding='utf-8') as f:
        start = False
        for file in f:
            file = file.strip()
            if not file:
                continue
            if file.startswith("<review") and not start:
                start = True
                if "label" in file:
                    labels.append(int(file.split('"')[3]))
                continue
            if start and file == r"</review>":
                start = False
                continue
            if start:
                reviews.append(file)
    if is_positive:
        labels = [1] * len(reviews)
    elif is_positive == False:
        labels = [0] * len(reviews)

    return reviews, labels


# 将数据写入变量
def process_file():
    train_pos_file = "sentiment/train.positive.txt"
    train_neg_file = "sentiment/train.negative.txt"
    test_comb_file = "sentiment/test.txt"

    # 读取文件部分，把具体的内容写入到变量里面
    train_pos_cmts, train_pos_lbs = load_data(train_pos_file, True)
    train_neg_cmts, train_neg_lbs = load_data(train_neg_file, False)
    train_comments = train_pos_cmts + train_neg_cmts
    train_labels = train_pos_lbs + train_neg_lbs
    test_comments, test_labels = load_data(test_comb_file)
    return train_comments, train_labels, test_comments, test_labels


train_comments, train_labels, test_comments, test_labels = process_file()

# 查看训练数据
print(len(train_comments), len(test_comments))

print(train_comments[3], train_labels[3])


# 数据预处理
def deal_text(text, stop_path):
    stopwords = set()
    with open(stop_path, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            stopwords.add(line.strip())

    text = re.sub('[!！]+', "!", text)
    text = re.sub('[?？]+', "?", text)
    text = re.sub("[a-zA-Z#$%&\'()*+,-./:;：<=>@，。★、…【】《》“”‘’[\\]^_`{|}~]+", " UNK ", text)
    text = re.sub(r"\d+", ' NUM ', text)
    text = re.sub(r"\s+", " ", text)
    text = " ".join([term for term in jieba.cut(text) if term and not term in stopwords])
    return text


train_comments_new = [deal_text(comment, "sentiment/stopwords.txt") for comment in train_comments]
test_comments_new = [deal_text(comment, "sentiment/stopwords.txt") for comment in test_comments]

print(train_comments_new[0], test_comments_new[0])

# 文本转向量
count_vector = CountVectorizer()
X_train = count_vector.fit_transform(train_comments_new)
y_train = train_labels
print(X_train)

X_test = count_vector.transform(test_comments_new)
y_test = test_labels
print(X_test)

print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))

# 训练并预测

clf = MultinomialNB(alpha=1.0, fit_prior=True)
# 利用朴素贝叶斯做训练
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy on test data: ", accuracy_score(y_test, y_pred))


