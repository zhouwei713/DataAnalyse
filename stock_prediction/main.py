# coding = utf-8
"""
@author: zhou
@time:2019/11/6 11:18
@File: main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
init_notebook_mode()
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


df = ts.get_hist_data('000001')
print(df)

df.dropna(axis=0 , inplace=True)
df.isna().sum()

# 按照时间升序排列
df.sort_values(by=['date'], inplace=True, ascending=True)
print(df.tail())

# 画K线图
trace = go.Ohlc(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])
data = [trace]
iplot(data, filename='simple_ohlc')

# 处理 label
num = 5 # 预测5天后的情况
df['label'] = df['close'].shift(-num) # 预测值
df.head(20)

feature = df.drop(['label', 'price_change', 'p_change'],axis=1)
print(feature.head())

X = feature.values
X = preprocessing.scale(X)
X = X[:-num]

df.dropna(inplace=True)
Target = df.label
y = Target.values

print(np.shape(X), np.shape(y))

# 将数据分为训练数据和测试数据
X_train, y_train = X[0:550, :], y[0:550]
X_test, y_test = X[550:, -51:], y[550:606]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

X_Predict = X[-num:]
Forecast = lr.predict(X_Predict)
print(Forecast)
print(y[-num:])
print(X_Predict)

trange = pd.date_range('2019-05-13', periods=num, freq='d')
Predict_df = pd.DataFrame(Forecast, index=trange)
Predict_df.columns = ['forecast']

# 将预测值添加到原始dataframe
df_new = ts.get_hist_data('000001')
# 按照时间升序排列
df_new.sort_values(by=['date'], inplace=True, ascending=True)
df_new.index = df_new.index.astype('datetime64[ns]')
df_concat = pd.concat([df_new, Predict_df], axis=1)

df_concat = df_concat[df_concat.index.isin(Predict_df.index)]

# 画预测值和实际值
df_concat['close'].plot(color='green', linewidth=1)
df_concat['forecast'].plot(color='orange', linewidth=3)
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()


