
"""
전남매울문화사업 사이트를 크롤링하기 위해 사용됨
http://munhwa.jndn.com/bbs/page.php?hid=write_his
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1920, 1080))

display.start()
options = webdriver.ChromeOptions()
service = Service(executable_path=ChromeDriverManager().install())

driver = webdriver.Chrome(service=service, options=options)

# driver.get("http://munhwa.jndn.com/bbs/page.php?hid=write_his")
# driver.find_element(By.XPATH, '/html/body/div[1]/div[4]/div/div/div/div/div[1]/label[2]').click()
# driver.find_element(By.XPATH, '//*[@id="thema_wrapper"]/div[4]/div/div/div/div/div[3]/h2[1]').text