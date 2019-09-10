# coding = utf-8
"""
@author: zhou
@time:2019/8/30 11:36
@File: main.py
"""


import requests
from bs4 import BeautifulSoup
import time
import os
from pyecharts import options as opts


def fire():
    page = 0
    for i in range(0, 450, 30):
        print("开始爬取第 %s 页" % page)
        url = 'https://movie.douban.com/celebrity/1011562/photos/?type=C&start={}&sortby=like&size=a&subtype=a'.format(i)
        res = requests.get(url).text
        data = get_poster_url(res)
        download_picture(data)
        page += 1
        time.sleep(1)


def get_poster_url(res):
    content = BeautifulSoup(res, "html.parser")
    data = content.find_all('div', attrs={'class': 'cover'})
    picture_list = []
    for d in data:
        plist = d.find('img')['src']
        picture_list.append(plist)
    return picture_list


def download_picture(pic_l):
    if not os.path.exists(r'picture'):
        os.mkdir(r'picture')
    for i in pic_l:
        pic = requests.get(i)
        p_name = i.split('/')[7]
        with open('picture\\' + p_name, 'wb') as f:
            f.write(pic.content)


if __name__ == '__main__':
    fire()
