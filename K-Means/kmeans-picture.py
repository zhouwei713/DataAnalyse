# coding = utf-8
"""
@author: zhou
@time:2019/10/10 18:41
@File: kmeans-picture.py
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn import preprocessing


img = Image.open('foot-small.jpg')
print(img)

width, height = img.size
data = []
for x in range(width):
    for y in range(height):
        r, g, b = img.getpixel((x, y))
        data.append([r, g, b])

# Min-Max 规范化
mm = preprocessing.MinMaxScaler()
img_data = mm.fit_transform(data)
img_mat = np.mat(img_data)
print(img_mat)

kmeans = KMeans(n_clusters=2)
kmeans.fit(img_mat)
label = kmeans.predict(img_mat)
label = label.reshape([width, height])

picture_mark = Image.new("L", (width, height))

for x in range(width):
    for y in range(height):
        picture_mark.putpixel((x, y), int(256/(label[x][y]+1))-1)

picture_mark.save("new-foot.jpg", "JPEG")

