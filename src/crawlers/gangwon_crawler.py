
"""
어린이강원일보 사이트를 크롤링하기 위해 사용됨
http://www.kidkangwon.co.kr/bbs/list.html?table=bbs_1&sc_category=4&sc_area=&sc_word=
"""

import requests
from bs4 import BeautifulSoup

GOODTEXT_PATH = "http://www.joungul.co.kr/joungul/index.asp"

req = requests.get(GOODTEXT_PATH)

html = req.text

soup = BeautifulSoup
