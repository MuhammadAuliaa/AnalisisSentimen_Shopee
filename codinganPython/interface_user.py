import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from function import mergedataFunction
import os
from function import preprocessingFunction
from sklearn.svm import SVC
from function import svmFunction
from imblearn.over_sampling import RandomOverSampler
import joblib
from function import scrapingFunction

# Modifikasi fungsi scrape_tokopedia_reviews
def scrape_tokopedia_reviews_user(url, jumlah_data, rating_min, rating_max):
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

            # Mengambil nama produk
            produk = 'Produk tidak ditemukan'
            if containersProduk:
                # Akses elemen pertama dalam containersProduk
                nama_produk = containersProduk[0].find('h1', attrs={'data-testid': 'lblPDPDetailProductName'})
                produk = nama_produk.text.strip() if nama_produk else 'Produk tidak ditemukan'
            
            # Mengambil nama pelanggan
            nama_pelanggan = container.find('span', attrs={'class': 'name'})
            pelanggan = nama_pelanggan.text.strip() if nama_pelanggan else 'Customer tidak ditemukan'
            
            # Mengambil rating
            rating_container = container.find('div', attrs={'data-testid': 'icnStarRating'})
            rating_label = rating_container['aria-label'] if rating_container else 'Tidak ada rating'
            rating = rating_mapping.get(rating_label, 'Tidak ada rating')

            # Filter data berdasarkan rentang rating yang diinginkan
            if rating_min <= rating <= rating_max:
                # Tambahkan data ke dalam list
                data.append((pelanggan, produk, review, rating))
                data_count += 1  # Tambahkan data yang sudah terkumpul

        # Klik tombol "Laman berikutnya" untuk memuat lebih banyak ulasan
        buttons = driver.find_elements(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']")
        if buttons:
            buttons[0].click()
            time.sleep(3)
        
    # Convert data ke DataFrame
    df = pd.DataFrame(data, columns=['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating'])
    
    # Tutup driver
    driver.close()
    
    return df

def clean(text):
  text = text.strip()
  text = text.lower()
  text = re.sub(r'[^a-zA-Z]+', ' ', text)
  return text

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
" tgl ":" tanggal ", " lg ":" lagi ", " bgt ":" banget ",' gua ':' saya ', '\n':' ', ' tpi ':' tapi ', ' standar ':' biasa ', ' standart ': ' biasa ', ' sdh ':' sudah ', ' n ':' dan ', ' gk ': ' tidak ', ' mengecwakan ':' mengecewakan ', ' d ':' di ', ' approved':' setuju', 'ademmmm ':'adem ', ' g ':' tidak ', ' gak ':' tidak ', 'gak ':'tidak ', ' cpt ':' cepat ', ' ku ':' aku ', ' design ':' desain ', ' purple ':' ungu ', 'bgus ':'bagus ', ' bgus ':' bagus ', ' stock ':' stok ', ' cumaa ':' hanya ', ' lmyan ':' lumayan ', ' gtu ':' gitu ', ' jatoh ':' jatuh ', ' koq ':' kok ', 'bnyk ':'banyak ', ' bnyk ':' banyak ', 'lucuuu ':'lucu ', ' lucuuu ':' lucu ', ' udh ':' udah ', ' mantaaaaaap ':' mantap ', ' check ':' cek ', ' mintiib ':' mantap ',
' bbrp ':' beberapa ', 'bbrp ':'beberapa ', 'sy ':'aku ', ' sy ':' saya ', ' pengiirman ':' pengiriman ', 'mantull ':'mantap betul ', 'bbrp ':'beberapa ', ' bbrp ':' beberapa ', ' brp ':' berapa ', 'brp ':'berapa ', ' makasiih ':' makasih ', 'makasiih ':'makasih ', 'napa ':'kenapa ', ' napa ':' kenapa ', ' jdnya ':' jadi ', 'jdnya ':'jadi ', ' sm ':' sama ', 'sm ':'sama ',
'nyobain ':'coba ',' nyobain ':' coba ', ' nyobain':' coba', 'kecewaaaaa ':'kecewa ',' kecewaaaaa ':' kecewa ', ' kecewaaaaa':' kecewa', 'sukak ':'suka ',' sukak ':' suka ', ' sukak':' suka', 'resp ':'respon ',' resp ':' respon ', ' resp':' respon', 'bangetttttttt ':'banget ',' bangetttttttt ':' banget ', ' bangetttttttt':' banget', 'tsb ':'tersebut ',' tsb ':' tersebut ', ' tsb':' tersebut', 'mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaap ':'mantap ',' mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaap ':' mantap ', ' mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaap':' mantap', 'cakepppp ':'bagus ',' cakepppp ':' bagus ', ' cakepppp':' bagus', 'keceee ':'keren ',' keceee ':' keren ', ' keceee':' keren', 'kece ':'keren ',' kece ':' keren ', ' kece':' keren', 'yng ':'yang ',' yng ':' yang ', ' yng':' yang', 'usa ':' ',' usa ':'  ', ' usa':' ', 'baguusss ':'bagus ',' baguusss ':' bagus ', ' baguusss':' bagus', 'disc ':'diskon ',' disc ':' diskon ', ' disc':' diskon', 'hehe ':' ',' hehe ':'  ', ' hehe':' ',
'bb ':'berat badan ',' bb ':' berat badan ', ' bb':' berat badan', 'tb ':'tinggi badan ',' tb ':' tinggi badan ', ' tb':' tinggi badan', 'kg ':'kilogram ',' kg ':' kilogram ', ' kg':' kilogram', 'bangettt ':'banget ',' bangettt ':' banget ', ' bangettt':' banget', 'jd ':'jadi ',' jd ':' jadi ', ' jd':' jadi', 'me ':'aku ',' me ':' aku ', ' me':' aku', 'gpp ':'gapapa ',' gpp ':' gapapa ', ' gpp':' gapapa', 'naikin ':'naik ',' naikin ':' naik ', ' naikin':' naik', 'lu ':'kamu ',' lu ':' kamu ', ' lu':' kamu', 'pny ':'punya ',' pny ':' punya ', ' pny':' punya', 'cepatt ':'cepat ',' cepatt ':' cepat ', ' cepatt':' cepat', 'banyakin ':'banyak ',' banyakin ':' banyak ', ' banyakin':' banyak', 'thx ':'makasih ',' thx ':' aku ', ' thx':' aku', 'dibeliin ':'beli ',' dibeliin ':' beli ', ' dibeliin':' beli', 'smpe ':'sampai ',' smpe ':' sampai ', ' smpe':' sampai', 'udh ':'udah ',' udh ':' udah ', ' udh':' udah', 'gmbr ':'gambar ',' gmbr ':' gambar ', ' gmbr':' gambar', 'bnykkk ':'banyak ',' bnykkk ':' banyak ', ' bnykkk':' banyak', 'dtg ':'datang ',' dtg ':' datang ', ' dtg':' datang', 'pcs ':'pieces ',' pcs ':' pieces ', ' pcs':' pieces', 'kermh ':'rumah ',' kermh ':' rumah ', ' kermh':' rumah', 'respononsif ':'responsif ',' respononsif ':' responsif ', ' respononsif':' responsif', 'seller ':'penjual ',' seller ':' penjual ', ' seller':' penjual', 'bhn ':'bahan ',' bhn ':' bahan ', ' bhn':' bahan',
'spt ':'seperti ',' spt ':' seperti ', ' spt':' seperti', 'lamaaa ':'lama ',' lamaaa ':' lama ', ' lamaaa':' lama', 'jgn ':'jangan ',' jgn ':' jangan ', ' jgn':' jangan', 'dimodif ':'modifikasi ',' dimodif ':' modifikasi ', ' dimodif':' modifikasi', ' pic ':' gambar ', ' tdi ':' tadi ', ' kyk ':' mirip ', ' seller ':' penjual ', ' skrg ':' sekarang ', ' nyesal ':' menyesal ', ' bagusss ':' bagus ', ' buy ':' beli ', ' kringet ':' keringat ', 'wkwk ':' ', ' wkwk ':' ', ' wkwk':' ', ' teball ':' tebal ', ' maksa ':' paksa ',
'plis ':'tolong ',' plis ':' tolong ', ' plis':' tolong', 'karenaa ':'karena ',' karenaa ':' karena ', ' karenaa':' karena', 'dsni ':'disini ',' dsni ':' disini ', ' dsni':' disini', 'beranrakan ':'berantakan ',' beranrakan ':' berantakan ', ' beranrakan':' berantakan', 'pakek ':'pakai ',' pakek ':' pakai ', ' pakek':' pakai', 'pdhl ':'padahal ',' pdhl ':' padahal ', ' pdhl':' padahal', ' kereeen ':' keren ', ' ttp ':' tetap ', ' bngt ':' banget ',
' lmyn ':' lumayan ', 'ujurannya ':'ukuran ', ' ujurannya ':' ukuran ', ' ujurannya':' ukuran', 'sblmnya ':'sebelum ', ' sblmnya ':' sebelum ', ' sblmnya':' sebelum', 'trnyta ':'ternyata ', ' trnyta ':' ternyata ', ' trnyta':' ternyata', ' hpus ':' hapus '}

def normalisasi(text):
  for i in norm:
    text = text.replace(i, norm[i])
  return text

def tokenisasi(text):
    return text.split() 
    
def stopword(text):
    stop_words = set(stopwords.words('indonesian'))
    words = text.split()
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

def filter_tokens_by_length(dataframe, column, min_words, max_words):
    # Tokenisasi kata
    words_count = dataframe[column].astype(str).apply(lambda x: len(x.split()))
    # Membuat filter untuk jumlah kata
    mask = (words_count >= min_words) & (words_count <= max_words)
    # Mengaplikasikan filter ke DataFrame
    df = dataframe[mask]
    return df

def stemming(text):
    stemmer = StemmerFactory().create_stemmer()
    text = ' '.join(text)
    stemmed_text = stemmer.stem(text)
    return stemmed_text

def labeling(rating):
    rating = str(rating)
    if rating == '4' or rating == '5':
       return 'Positif'
    else :
       return 'Negatif'

with st.sidebar:
    selected = option_menu("Main Menu", ["User", "Dashboard", "Scraping", "Merge Data", 'Dataset', 'Preprocessing', "Visualization", "Support Vector Machine", "IndoBert", "Testing"], 
        icons=['house','house', 'gear', 'book', 'pen', 'pen', 'book', 'kanban','activity', 'activity', 'cloud-upload' ], menu_icon="cast", default_index=0)
    selected

if selected == 'User':
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image('img/tokopedia.png', width=80)
    with col2:
        st.title("Scraping Data")

    url = st.text_input("Masukkan URL produk:")
    jumlah_data = st.number_input("Masukkan jumlah data yang ingin diambil (*Max 50 baris) :", min_value=1, step=1, value=10)

    if st.button("Mulai Scraping"):
        jumlah_data_rating_low = int(jumlah_data * 0.3)  # 30% untuk rating 1-3
        jumlah_data_rating_high = jumlah_data - jumlah_data_rating_low  # 70% untuk rating 4-5
        
        # Mengambil data rating 1-3
        data_low = scrape_tokopedia_reviews_user(url, jumlah_data_rating_low, rating_min=1, rating_max=3)
        
        # Mengambil data rating 4-5
        data_high = scrape_tokopedia_reviews_user(url, jumlah_data_rating_high, rating_min=4, rating_max=5)
        
        # Menggabungkan data
        data_combined = pd.concat([data_low, data_high]).reset_index(drop=True)
        
        # =====================================
        # Preprocessing
        data_combined['Ulasan'] = data_combined['Ulasan'].fillna('')
        data_combined['Ulasan'] = data_combined['Ulasan'].apply(clean)
        data_combined['Ulasan'] = data_combined['Ulasan'].apply(normalisasi)
        data_combined['Ulasan'] = data_combined['Ulasan'].apply(stopword)
        data_combined['Ulasan'] = data_combined['Ulasan'].apply(tokenisasi)

        min_words = 3
        max_words = 100
        data_combined = filter_tokens_by_length(data_combined, 'Ulasan', min_words, max_words)

        data_combined['Ulasan'] = data_combined['Ulasan'].apply(stemming)
        data_combined['Sentimen'] = data_combined['Rating'].apply(labeling)
        st.success("Proses data berhasil!")
        st.write(data_combined[['Ulasan', 'Sentimen']])

        # Segmentasi dan Visualisasi
        segmentation_keywords = {
            'bahan': ['tipis', 'tebal', 'lembut', 'keras', 'kasar', 'rapih', 'rapi', 'pendek', 'adem', 'nyaman', 'jahit', 'halus', 'gerah', 'relaxing', 'baju', 'model', 'celana', 'nama', 'transparan', 'badan', 'sayap'],
            'kualitas': ['rusak', 'sesuai', 'bagus', 'jelek', 'berkualitas', 'keringat', 'sobek', 'aneh', 'foto', 'gambar', 'keren', 'mantap', 'kecil', 'label', 'ngetat', 'ketat', 'pict', 'fashion', 'bolong', 'style', 'sederhana'],
            'warna': ['cerah', 'pudar', 'gelap', 'putih', 'hitam', 'warna', 'biru', 'soft', 'navy', 'pink']
        }

        segmentation_order = list(segmentation_keywords.keys())
        
        fig, axes = plt.subplots(nrows=1, ncols=len(segmentation_order), figsize=(20, 6), sharey=True)
        st.header('Visualisasi Aspek :')

        for idx, segment in enumerate(segmentation_order):
            keywords = segmentation_keywords[segment]
            segment_data = data_combined[data_combined['Ulasan'].str.contains('|'.join(keywords), case=False)]

            if not segment_data.empty:
                sentiment_counts = segment_data['Sentimen'].value_counts()
                total_counts = sentiment_counts.sum()
                sentiment_df = pd.DataFrame({
                    'Sentimen': sentiment_counts.index,
                    'Count': sentiment_counts.values,
                    'Percentage': sentiment_counts.values / total_counts * 100
                })

                sns.barplot(ax=axes[idx], x='Sentimen', y='Percentage', data=sentiment_df, palette={'Positif': '#037ffc', 'Negatif': '#fc0324'}, dodge=False)
                axes[idx].set_title(f"Aspek {segment}")
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
                if idx == 0:
                    axes[idx].set_ylabel("Percentage")

                # Add percentage labels above bars
                for p in axes[idx].patches:
                    height = p.get_height()
                    axes[idx].text(
                        p.get_x() + p.get_width() / 2.,
                        height + 0.5,
                        f'{height:.1f}%',
                        ha='center',
                        va='bottom'
                    )
            else:
                st.warning(f"No data found for aspek: {segment}")

        fig.tight_layout(pad=1.0)  # Add spacing between rows
        st.pyplot(fig)
           
