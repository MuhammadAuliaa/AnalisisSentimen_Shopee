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
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

selected = option_menu(None, ["Crawling", "Dataset", 'Preprocessing', 'TF-IDF' ,'Machine Learning', 'Prediction'], 
    icons=['cloud-upload', "archive", 'gear', 'activity', 'kanban', 'kanban'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

norm= {" dgn " : " dengan ", ' seller ': ' penjual ',' service ':' pelayanan ', ' tp ':' tapi ', ' recommended ':' rekomendasi ', ' kren ':' keren ', ' kereen ':' keren ', ' mantab ': ' keren ',' matching ':' sesuai ','happy':' senang ','original': 'asli ','ori':'asli ', "trusted" : "terpercaya", "angjaaaassss":"keren", " gue ": " saya ", "bgmn ":" bagaimana ", ' tdk':' tidak ', ' blum ':' belum ', 'mantaaaaaaaappp':' bagus ', 'mantaaap':'bagus ', ' josss ':' bagus ', ' thanks ': ' terima kasih ', 'fast':' cepat ', ' dg ':' dengan ', 'trims':' terima kasih ', 'brg':' barang ', 'gx':' tidak ', ' dgn ':' dengan ', ' recommended':' rekomen ', 'recomend':' rekomen ', 'good':' bagus ', " dgn " : " dengan ", " gue ": " saya ", " dgn ":" dengan ", "bgmn ":" bagaimana ", ' tdk':' tidak ', 
' blum ':' belum ', "quality":"kualitas", 'baguss':'bagus', 'overall' : 'akhirnya', 'mantaaaaaaaappp':' bagus ', ' josss ':' bagus ', ' thanks ': ' terima kasih ', 'fast':' cepat ', 
 'trims':' terima kasih ', 'brg':' barang ', 'gx':' tidak ', ' dgn ':' dengan ', ' real ': ' asli ', ' bnb ': ' baru ' ,
' recommended':' rekomen ', 'recomend':' rekomen ', 'good':'bagus',
'eksis ':'ada ', 'beenilai ':'bernilai ', ' dg ':' dengan ', ' ori ':' asli ', ' setting ':' atur ', " free ":" gratis ",
' yg ':' yang ', 't4 ':'tempat', ' awat ':' awet', ' mantep ':' bagus ', 'mantapp':'bagus', 
'kl ':'kalo', ' k ':' ke ', 'plg ':'pulang ', 'ajah ':'aja ', 'bgt':'banget', 'lbh ':'lebih', 'ayem':'tenang','dsana ':'disana ', 'lg':' lagi',
'pas ':'saat ', ' bnib ': ' baru ', 
' nggak ':' tidak ', 'karna ':'karena ', 'utk ':'untuk ',
' dn ':' dan ', ' mlht ':' melihat ', ' pd ':' pada ', 'mndngr ':'mendengar ', 'crita':'cerita', ' dpt ':' dapat ', ' mksh ':' terima kasih ', ' sellerrrr':' penjual', 'ori ':'asli ', ' new ':' baru ',
'sejrh':'sejarah', 'mnmbh ':'menambah ', 'sayapun':'saya', 'thn ':'tahun ', 'good':'bagus', ' awettt':' awet',
'halu ':'halusinasi ', ' nyantai ':' santai ', 'plus ':'dan ',
' ayang ':' sayang ', ' Rekomendded ':' direkomendasikan ', ' now ': ' sekarang ', 'slalu ':'selalu ', 'photo ': 'foto ', 'slah ':'salah ', 'krn':'karena', ' ga ':' tidak ', 'ok ':'oke ', ' meski':' mesti', ' para ':'parah', ' nawarin':' menawari', 'socmed':'sosial media',
' sya ':' saya ', 'siip':'bagus', ' bny ':' banyak ', ' tdk ':' tidak ', ' byk ':' banyak ', 
' pool ':' sekali ', " pgn ":" ingin ", " gue ":" saya ", " bgmn ":" bagaimana ", " ga ":" tidak ", 
" gak ":" tidak ", " dr ":" dari ", " yg ":" yang ", " lu ":" kamu ", " sya ":" saya ", 
" lancarrr ":" lancar ", " kayak ":" seperti ", " ngawur ":" sembarangan ", " k ":" ke ", 
" luasss ":" luas ", " sy ":" saya ", " thn ":" tahun ", " males ":" malas ",
" tgl ":" tanggal ", " lg ":" lagi ", " bgt ":" banget ",' gua ':' saya ', '\n':' ', ' tpi ':' tapi ', ' standar ':' biasa ', ' standart ': ' biasa ', ' sdh ':' sudah ', ' n ':' dan ', ' gk ': ' tidak ', ' mengecwakan ':' mengecewakan ', ' d ':' di '}

def normalisasi(text):
  for i in norm:
    text = text.replace(i, norm[i])
  return text

def clean(text):
  text = text.strip()
  text = text.lower()
  text = re.sub(r'[^a-zA-Z]+', ' ', text)
  return text

def labeling(rating):
    if rating == '4' or rating == '5':
        return 'Positif'
    else:
        return 'Negatif'

def tokenisasi(text):
    return text.split() 
    
def stopword(text):
    stop_words = set(stopwords.words('indonesian'))
    words = text.split()
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

def stemming(text):
    stemmer = StemmerFactory().create_stemmer()
    text = ' '.join(text)
    stemmed_text = stemmer.stem(text)
    return stemmed_text

if selected == 'Crawling':
    col1, col2 = st.columns([1,8])
    with col1:
        st.image('img/tokopedia.png', width=80)
    with col2:
        st.title("Crawling Data")

    # https://www.tokopedia.com/homedoki/review
    # https://www.tokopedia.com/pengrajincom/review 
    url = st.text_input("URL")
    tombol = st.button("Crawling")  
    if tombol :
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

            # Pastikan jika data yang diinginkan sudah terpenuhi
            if data_count >= jumlah_data:
                break

            # FIX ERROR!
            buttons = driver.find_elements(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']")
            if buttons:
                buttons[0].click()
                time.sleep(3)

        df = pd.DataFrame(data, columns=['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating'])   
        df.to_csv('dataScrapingHanaShop/Tokopedia.csv', index=False)
        driver.close()
        # Mengubah tipe data 'Rating' & memanggil attribut tertentu
        showDf = pd.read_csv('dataScrapingHanaShop/dataHanaShop_10produk.csv', dtype={'Rating': 'object'})
        showDf = showDf[['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating']]

        # Menampilkan data
        st.write(showDf)

        # Pengkondisian Alert Crawling
        jumlah_data = len(showDf)
        if jumlah_data > 0:
            st.success(f"Crawling {jumlah_data} Baris Data Berhasil!")
        else:
            st.warning("Crawling Data Gagal")

if selected == 'Dataset':
    st.title("Dataset Tokopedia :")
    try:
       df = pd.read_csv("dataScrapingHanaShop/dataHanaShop_10produk.csv", dtype={'Rating':'object'})
       df = df[['Nama Pelanggan','Produk','Ulasan', 'Rating']]
       st.dataframe(df)
    except FileNotFoundError:
       st.write("File not found, please check the file path...")
