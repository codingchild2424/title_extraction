
"""
문학광장 사이트를 크롤링하기 위해 사용됨
https://teen.munjang.or.kr/archives/category/write/life
"""

import requests
from bs4 import BeautifulSoup
import os
from urllib.request import urlopen
from urllib.request import Request
from tqdm import tqdm

import pandas as pd


# 폴더 생성
save_folder_path = "/workspace/home/uglee/Projects/title_extraction/datasets/raw_datasets/teen"

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Folder made")
    except OSError:
        print("Error")

createFolder(save_folder_path)

# https://teen.munjang.or.kr/archives/category/write/life/page/2

# URL
TEEN_URL_LIST = [
    'https://teen.munjang.or.kr/archives/category/write/life/page/' + \
    str(i + 1)
    for i in range(358)
]

'''
[공지] 빼고 크롤링하기
'''

#essay_urls = []

essay_link_list = []

# 번호별 페이지
for idx, teen_url in enumerate(tqdm(TEEN_URL_LIST)):

    # gangwon_url 은 각 페이지별 url

    # page 하나 가져오기
    req = Request(teen_url, headers={'User-Agent': 'Mozila/5.0'})
    webpage = urlopen(req)
    soup = BeautifulSoup(webpage, 'html.parser')

    # 한 페이지 내에서만 찾기
    for i, tag in enumerate(soup.find_all('a', attrs={'rel': 'bookmark'})):

        if tag.text in "[공지]":
            continue

        essay_link = tag.attrs["href"]

        essay_link_list.append(essay_link)

print(essay_link_list)

title_list = []
content_list = []

# 각 essay 별로 접근
for idx, essay_link in enumerate(essay_link_list):

    req = Request(essay_link, headers={'User-Agent': 'Mozila/5.0'})
    webpage = urlopen(req)
    soup = BeautifulSoup(webpage, 'html.parser')

    title = soup.find('h1', 'entry-title').text
    title_list.append(title)

    content = soup.find('h1', 'entry-content').text
    content_list.append(content)

    '''
    확인 후 아래 print if 지우기
    '''

    print("content", content)

    if idx == 0:
        break

print(title_list)
print(content_list)

# title과 content 를 \t으로 합쳐서 tsv로 각각 저장해서 pre_datasets로 넘기기

save_pre_folder_path = "/workspace/home/uglee/Projects/title_extraction/datasets/pre_datasets"

for title, content in zip(title_list, content_list):

    save_pre_file_path = os.path.join(save_pre_folder_path, title + ".tsv")

    title_content = pd.DataFrame(title + "\t" + content)

    title_content.to_csv(save_pre_file_path, header=False, index=False)

    




    # page에서 각 작품별 링크 가져오기
    # for td_tag in soup.find_all('td', 'bbs-list-title bbs-skin-width-large'):

        # 장르가 [수필] 이나 [산문] 일때만 url 저장
        # if td_tag.find('small').text == '[수필]' or td_tag.find('small').text == '[산문]':
        #     essay_url = "http://www.kidkangwon.co.kr/bbs/" + td_tag.find('a')['href']
        #     #essay_urls.append(essay_url)
            
        #     essay_req = Request(essay_url, headers={'User-Agent': 'Mozila/5.0'})
        #     essay_webpage = urlopen(essay_req)
        #     essay_soup = BeautifulSoup(essay_webpage, 'html.parser')

        #     title = essay_soup.find('div', 'header-title').text
        #     content = essay_soup.find('article', 'content').text

        #     title = title.replace("/", "")

        #     title_content = title + '\t' + content

        #     # tsv로 파일 저장하기
        #     save_file_path = os.path.join(save_folder_path, title + ".tsv")
        #     if not os.path.exists(save_file_path):
        #         f = open(save_file_path, 'w')
        #         f.write(title_content)
        #         f.close()

            

# print(essay_urls)

# title_list = []
# content_list = []

# # 각 url 별로 제목과 본문 추출하기
# for essay_url in tqdm(essay_urls):
#     req = Request(essay_url, headers={'User-Agent': 'Mozila/5.0'})
#     webpage = urlopen(req)
#     soup = BeautifulSoup(webpage, 'html.parser')

#     title = soup.find('div', 'header-title').text
#     content = soup.find('article', 'content').text

#     title_list.append(title)
#     content_list.append(content)


# save_folder_path = "/workspace/home/uglee/Projects/title_extraction/datasets/raw_datasets/gangwon_folder"

# import pandas as pd

# df = pd.DataFrame({
#     'title': title,
#     'content': content
# })

# df.to_csv(
#     os.join(save_folder_path, 'gangwon_dataset.csv'),
#     sep='\t',
#     encoding='utf-8'
# )



