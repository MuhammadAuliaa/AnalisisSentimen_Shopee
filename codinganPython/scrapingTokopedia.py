import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By

def scrape_tokopedia_reviews(url, jumlah_data):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    rating_mapping = {
        'bintang 5': 5,
        'bintang 4': 4,
        'bintang 3': 3,
        'bintang 2': 2,
        'bintang 1': 1
    }

    data = []
    data_count = 0 

    while data_count < jumlah_data:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.findAll('article', attrs={'class': 'css-ccpe8t'})

        for container in containers:
            if data_count >= jumlah_data:
                break  

            review_container = container.find('span', attrs={'data-testid': 'lblItemUlasan'})
            review = review_container.text.strip() if review_container else 'Tidak ada ulasan'

            nama_produk = container.find('p', attrs={'class': 'e1qvo2ff8'})
            produk = nama_produk.text.strip() if nama_produk else 'Produk tidak ditemukan'

            nama_pelanggan = container.find('span', attrs={'class': 'name'})
            pelanggan = nama_pelanggan.text.strip() if nama_pelanggan else 'Customer tidak ditemukan'

            rating_container = container.find('div', attrs={'data-testid': 'icnStarRating'})
            rating_label = rating_container['aria-label'] if rating_container else 'Tidak ada rating'
            rating = rating_mapping.get(rating_label, 'Tidak ada rating')

            data.append((pelanggan, produk, review, rating))
            data_count += 1  

        if data_count >= jumlah_data:
            break

        next_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']")
        if next_button:
            next_button.click()
            time.sleep(3)
        else:
            break

    df = pd.DataFrame(data, columns=['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating'])
    df.to_csv('dataTokopedia.csv', index=False)
    driver.close()

url = "https://www.tokopedia.com/hanafashionshop/review"
jumlah_data = 100
scrape_tokopedia_reviews(url, jumlah_data)
