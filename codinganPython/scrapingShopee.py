import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

url = "https://shopee.co.id/Hana-Fashion-Arisha-Casual-Long-Dress-Wanita-CD047-2-i.171615412.22638832543?xptdk=e368175c-8f33-49cf-beab-a7ced757b7ff"
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
driver.get(url)

soup = BeautifulSoup(driver.page_source, "html.parser")
containers = soup.findAll('div', attrs={'class':'shopee-product-rating'})
print(len(containers))
print(containers)