import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By

# Daftar input 10 nama produk dari pengguna
nama_produk_list = [
    "HanaFashion - Danise Short Pants Celana Pendek Wanita - SP061 - yellow green, XL",
    "Hanafe Crop Top Hana Fashion - Light Grey, M",
    "HanaFashion - Lucia Basic Casual Cardigan Outer Simple Wanita - SB136 - MC Polka, S",
    "HanaFashion - Woori Blue Jeans Long Pants - JLP043",
    "HanaFashion - Felecia Basic Outer Atasan Wanita - SB144 - Mustard, S",
    "HanaFashion - Charlotte Short Pants Celana Pendek Wanita - SP050 - Purple, S",
    "HanaFashion - Martina Basic - Kaos T Shirt Crop Top Murah - TS177 - Khaky, XXL",
    "HanaFashion - Tiara Short Pants Celana Pendek Wanita - SP041 - Pink, S",
    "HanaFashion - Sita Crop Tank Top Atasan Wanita - TT007 - Khaky, XL",
    "HanaFashion - Kendall Crop Tank Top Wanita - TT114 - Royal Blue, S"
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
    # Kamus untuk melacak jumlah ulasan yang disimpan per produk
    produk_ulasan_count = {produk: 0 for produk in nama_produk_list}

    # Loop untuk melakukan scraping
    while True:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.findAll('article', attrs={'class': 'css-ccpe8t'})

        for container in containers:
            # Dapatkan nama produk
            nama_produk = container.find('p', attrs={'class': 'e1qvo2ff8'})
            produk = nama_produk.text.strip() if nama_produk else 'Produk tidak ditemukan'

            # Periksa apakah produk dalam daftar input pengguna
            if produk not in nama_produk_set:
                continue

            # Periksa apakah produk sudah memiliki 70 ulasan
            if produk_ulasan_count[produk] >= 70:
                continue
            
            # Tambahkan produk ke set products_scraped
            produk_ulasan_count[produk] += 1

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

        # Periksa apakah semua produk sudah memiliki 70 ulasan
        if all(count >= 70 for count in produk_ulasan_count.values()):
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
