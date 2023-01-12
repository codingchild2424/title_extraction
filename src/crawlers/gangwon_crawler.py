
"""
어린이강원일보 사이트를 크롤링하기 위해 사용됨
http://www.kidkangwon.co.kr/bbs/list.html?table=bbs_1&sc_category=4&sc_area=&sc_word=
"""

import requests
from bs4 import BeautifulSoup
import os
from urllib.request import urlopen
from urllib.request import Request
from tqdm import tqdm


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

# http://www.kidkangwon.co.kr/bbs/list.html?page=2&total=200473&table=bbs_1&sc_area=&sc_word=&category=

# URL
GANGWON_URL_LIST = [
    'http://www.kidkangwon.co.kr/bbs/list.html?page=' + \
    str(i + 1)  + \
    '&total=200473&table=bbs_1&sc_area=&sc_word=&category=' \
    for i in range(10024)
]

'''
모든 장르가 섞여있음
각 작품별 url 가져올때, 각 작품이 [수필]인지 확인해서 데이터 수집하기
'''

essay_urls = []

for gangwon_url in tqdm(GANGWON_URL_LIST):

    # gangwon_url 은 각 페이지별 url

    # page 하나 가져오기
    req = Request(gangwon_url, headers={'User-Agent': 'Mozila/5.0'})
    webpage = urlopen(req)
    soup = BeautifulSoup(webpage, 'html.parser')

    # page에서 각 작품별 링크 가져오기
    for td_tag in soup.find_all('td', 'bbs-list-title bbs-skin-width-large'):

        # 장르가 [수필] 이나 [산문] 일때만 url 저장
        if td_tag.find('small').text == '[수필]' or td_tag.find('small').text == '[산문]':
            essay_url = "http://www.kidkangwon.co.kr/bbs/" + td_tag.find('a')['href']
            essay_urls.append(essay_url)

print(essay_urls)

title_list = []
content_list = []

# 각 url 별로 제목과 본문 추출하기
for essay_url in tqdm(essay_urls):
    req = Request(essay_url, headers={'User-Agent': 'Mozila/5.0'})
    webpage = urlopen(req)
    soup = BeautifulSoup(webpage, 'html.parser')

    title = soup.find('div', 'header-title').text
    content = soup.find('article', 'content').text

    title_list.append(title)
    content_list.append(content)


# save_folder_path = "/workspace/home/uglee/Projects/title_extraction/datasets/raw_datasets/gangwon_folder"

import pandas as pd

df = pd.DataFrame({
    'title': title,
    'content': content
})

df.to_csv(
    os.join(save_folder_path, 'gangwon_dataset.csv'),
    sep='\t',
    encoding='utf-8'
)



