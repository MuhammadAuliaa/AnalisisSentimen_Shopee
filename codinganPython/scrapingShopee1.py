from requests import get
from bs4 import BeautifulSoup

url = "https://shopee.co.id/Hana-Fashion-Arisha-Casual-Long-Dress-Wanita-CD047-2-i.171615412.22638832543?xptdk=e368175c-8f33-49cf-beab-a7ced757b7ff"
response= get(url)
soup=BeautifulSoup(response.text,'html.parser')
print(soup)