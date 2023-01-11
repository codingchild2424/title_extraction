
"""
좋은글 사이트를 크롤링하기 위해 사용됨
http://munhwa.jndn.com/bbs/page.php?hid=write_his
"""

import requests
from bs4 import BeautifulSoup

GOODTEXT_PATH = "http://www.joungul.co.kr/joungul/index.asp"

req = requests.get(GOODTEXT_PATH)

html = req.text

soup = BeautifulSoup
