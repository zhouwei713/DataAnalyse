# coding = utf-8
"""
@author: zhou
@time:2019/10/10 18:41
@File: kmeans-picture.py
"""

import matplotlib.pyplot as plt
from numpy import reshape
from sklearn.cluster import KMeans
from copy import deepcopy
from PIL import Image

img = plt.imread('foot-small.jpg')

pixel = reshape(img, (img.shape[0] * img.shape[1], 3))
pixel_new = deepcopy(pixel)

print(img.shape)

model = KMeans(n_clusters=5)
labels = model.fit_predict(pixel)
palette = model.cluster_centers_

for i in range(len(pixel)):
    pixel_new[i, :] = palette[labels[i]]

new_pic = reshape(pixel_new, (img.shape[0], img.shape[1], 3))
images = Image.fromarray(new_pic)
images.save("foot-new.jpg")
