import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from streamlit_option_menu import option_menu
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from tabulate import tabulate
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
import time

# Fungsi scrape_tokopedia_reviews
def scrape_tokopedia_reviews(url, jumlah_data, file_path):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # Mapping untuk rating
    rating_mapping = {
        'bintang 5': 5,
        'bintang 4': 4,
        'bintang 3': 3,
        'bintang 2': 2,
        'bintang 1': 1
    }

    # Menyimpan data ulasan
    data = []
    data_count = 0  # Jumlah data yang sudah dikumpulkan

    while data_count < jumlah_data:
        # Membuat soup dari halaman sumber
        soup = BeautifulSoup(driver.page_source, "html.parser")
        # Mengambil kontainer ulasan
        containers = soup.findAll('article', attrs={'class': 'css-72zbc4'})
        # Mengambil kontainer produk
        containersProduk = soup.findAll('div', attrs={"class":"css-1fogemr"})

        for container in containers:
            if data_count >= jumlah_data:
                break  # Jika jumlah data yang diinginkan sudah terpenuhi, hentikan loop

            # Mengambil review dari kontainer
            review_container = container.find('span', attrs={'data-testid': 'lblItemUlasan'})
            review = review_container.text.strip() if review_container else 'Tidak ada ulasan'

            # Pastikan containersProduk tidak kosong sebelum mencoba mengakses elemen pertama
            if containersProduk:
                # Akses elemen pertama dalam containersProduk
                nama_produk = containersProduk[0].find('h1', attrs={'data-testid': 'lblPDPDetailProductName'})
                produk = nama_produk.text.strip() if nama_produk else 'Produk tidak ditemukan'
            else:
                produk = 'Produk tidak ditemukan'

            nama_pelanggan = container.find('span', attrs={'class': 'name'})
            pelanggan = nama_pelanggan.text.strip() if nama_pelanggan else 'Customer tidak ditemukan'

            # Mengambil rating
            rating_container = container.find('div', attrs={'data-testid': 'icnStarRating'})
            rating_label = rating_container['aria-label'] if rating_container else 'Tidak ada rating'
            rating = rating_mapping.get(rating_label, 'Tidak ada rating')

            # Tambahkan data ke dalam list
            data.append((pelanggan, produk, review, rating))
            data_count += 1  # Tambahkan data yang sudah terkumpul

        if data_count >= jumlah_data:
            break

        buttons = driver.find_elements(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']")
        if buttons:
            buttons[0].click()
            time.sleep(3)

    df = pd.DataFrame(data, columns=['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating'])
    df.to_csv(file_path + ".csv", index=False)
    driver.close()