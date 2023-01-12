
"""
좋은글 사이트를 크롤링하기 위해 사용됨
http://www.joungul.co.kr/joungul/index.asp
"""

import requests
from bs4 import BeautifulSoup

from urllib.request import urlopen
from urllib.request import Request
from tqdm import tqdm

import os

# 각 글의 유형별 url 리스트 구현
GOODTEXT_URL_LIST = [ 
    'http://www.joungul.co.kr/impression/impression' + \
    str(i + 1) + \
    '/list.asp' \
    for i in range(7)
    ]

#http://www.joungul.co.kr/impression/impression1/list.asp
# 
# ?GotoPage=

# 폴더 생성
save_folder_path = "/workspace/home/uglee/Projects/title_extraction/datasets/raw_datasets/good_text_folder"

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Folder made")
    except OSError:
        print("Error")

createFolder(save_folder_path)

# 좋은글 페이지에서 글 가져오기
title_contents = []

# 글 유형별
for goodtext_url in GOODTEXT_URL_LIST:

    # 전체 글의 수 가져오기
    page_num_req = Request(goodtext_url, headers={'User-Agent': 'Mozila/5.0'})
    page_num_webpage = urlopen(page_num_req)
    page_num_soup = BeautifulSoup(page_num_webpage, 'html.parser')
    page_num = page_num_soup.find_all('option')[len(page_num_soup.find_all('option')) - 1].text

    # 전체 글의 수만큼 반복
    for i in tqdm(range(int(page_num))):
        # 각 페이지 별 소스 추출
        goodtext_page_url = goodtext_url + '?GotoPage=' + str(i + 1)
        req = Request(goodtext_page_url, headers={'User-Agent': 'Mozila/5.0'})
        webpage = urlopen(req)
        soup = BeautifulSoup(webpage, 'html.parser')

        # 각 페이지에서 a태그만 가져와서 그 안에 있는 글 추출하기
        for tag in soup.find_all('a'): 
            # 만약 a 태그 안에서 title attribute가 있다면 가져오기
            if tag.has_attr("title"): 
                title_contents.append(tag["title"])

print("title_contents", title_contents)

for idx, title_content in enumerate(title_contents):

    save_path = save_folder_path + '/good_text' + str(idx) + '.txt'

    f = open(save_path, 'w', encoding='utf-8')
    f.write(title_content)
    f.close()





