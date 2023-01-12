
"""
어린이강원일보 사이트를 크롤링하기 위해 사용됨
http://www.kidkangwon.co.kr/bbs/list.html?table=bbs_1&sc_category=4&sc_area=&sc_word=
"""

import requests
from bs4 import BeautifulSoup
import os
from urllib.request import urlopen
from urllib.request import Request

# 폴더 생성
save_folder_path = "/workspace/home/uglee/Projects/title_extraction/datasets/raw_datasets/gangwon_folder"

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Folder made")
    except OSError:
        print("Error")

createFolder(save_folder_path)

# URL
GANGWON_URL_LIST = [
    'http://www.kidkangwon.co.kr/bbs/list.html?page=' + \
    str(i + 1)  + \
    '&total=3630&table=bbs_1&sc_area=&sc_word=&category=' \
    for i in range(182)
]

TEST_URL = GANGWON_URL_LIST[0]

req = Request(TEST_URL, headers={'User-Agent': 'Mozila/5.0'})

webpage = urlopen(req)

soup = BeautifulSoup(webpage, 'html.parser')

for td_tag in soup.find_all('td', 'bbs-list-title bbs-skin-width-large'):

    print(td_tag)

    # if tag.has_attr("href"):
    #     print(tag['href'])


