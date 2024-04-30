import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By

# Terima input 5 nama produk dari pengguna
nama_produk_list = [
    "HanaFashion - Kharen T-shirt Knit / Kaos Crop Wanita Korea - TS478",
    "HanaFashion - Bona T-shirt Wanita / Kaos Wanita Gaya Korea - TS421",
    "HanaFashion - Anya Mini Skirt / Rok Pendek Wanita Murah - SK171",
    "HanaFashion - Woori Blue Jeans Long Pants - JLP043",
    "Hana Fashion - Nagita Set Strap Midi Dress With Cardigan - ST155"
]

# Mengubah daftar produk ke dalam set untuk pencarian cepat
nama_produk_set = set(nama_produk_list)

# URL situs yang akan di-scrape
url = "https://www.tokopedia.com/hanafashionshop/review"

if url:
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
    products_scraped = set()  # Untuk melacak produk yang sudah di-scrape

    # Loop untuk melakukan scraping
    while len(products_scraped) < 5:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.findAll('article', attrs={'class': 'css-ccpe8t'})

        for container in containers:
            # Dapatkan nama produk
            nama_produk = container.find('p', attrs={'class': 'e1qvo2ff8'})
            produk = nama_produk.text.strip() if nama_produk else 'Produk tidak ditemukan'

            # Periksa apakah produk dalam daftar input pengguna
            if produk not in nama_produk_set:
                continue

            # Periksa apakah produk sudah di-scrape
            if produk in products_scraped:
                continue
            
            # Tambahkan produk ke set products_scraped
            products_scraped.add(produk)

            # Dapatkan ulasan dan informasi pelanggan
            review_container = container.find('span', attrs={'data-testid': 'lblItemUlasan'})
            review = review_container.text.strip() if review_container else 'Tidak ada ulasan'

            nama_pelanggan = container.find('span', attrs={'class': 'name'})
            pelanggan = nama_pelanggan.text.strip() if nama_pelanggan else 'Customer tidak ditemukan'

            # Dapatkan rating
            rating_container = container.find('div', attrs={'data-testid': 'icnStarRating'})
            rating_label = rating_container['aria-label'] if rating_container else 'Tidak ada rating'
            rating = rating_mapping.get(rating_label, 'Tidak ada rating')

            # Simpan data yang di-scrape
            data.append((pelanggan, produk, review, rating))

            # Jika sudah 5 produk yang serupa, hentikan scraping
            if len(products_scraped) >= 5:
                break

        # Lanjutkan ke halaman berikutnya
        next_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']")
        if next_button:
            next_button.click()
        time.sleep(3)

    # Simpan data yang di-scrape ke file CSV
    df = pd.DataFrame(data, columns=['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating'])
    df.to_csv('Tokopedia.csv', index=False)
    driver.close()
