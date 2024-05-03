import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
import seaborn as sns
from streamlit_option_menu import option_menu
import pandas as pd
import time
from function import scrapingFunction
from function import preprocessingFunction

with st.sidebar:
    selected = option_menu("Main Menu", ["Crawling", 'Dataset', 'Preprocessing'], 
        icons=['house', 'gear', 'book', 'pen', 'pen', 'book', 'kanban','activity', 'activity', 'cloud-upload' ], menu_icon="cast", default_index=0)
    selected

if selected == "Crawling":
    col1, col2 = st.columns([1,8])
    with col1:
        st.image('img/tokopedia.png', width=80)
    with col2:
        st.title("Crawling Data")

    url = st.text_input("Masukkan URL produk:")
    nama_file = st.text_input("Masukkan nama file hasil scraping data:")
    jumlah_data = st.number_input("Masukkan jumlah data yang ingin diambil:", min_value=1, step=1, value=10)
    folder_path = "dataScrapingHanaShop/"
    file_path = folder_path + nama_file

    if st.button("Mulai Scraping"):
        scrapingFunction.scrape_tokopedia_reviews(url, jumlah_data, file_path)
        st.warning(f"Data telah disimpan ke: {file_path}.csv")

if selected == "Dataset":
    st.title("Dataset Tokopedia :")
    uploaded_file = st.file_uploader("Upload .CSV file", type=["csv"])
    if uploaded_file is not None:
        try :
            df = pd.read_csv(uploaded_file, dtype={"Rating":"object"}, index_col=0)
            st.dataframe(df)
        except pd.errors.EmptyDataError:
            st.write("File is empty, please check your input.")
        except pd.errors.ParserError:
            st.write("Invalid data format, please check your input.")

elif selected == "Preprocessing":
    st.title("Preprocessing Data")
    uploaded_file = st.file_uploader("Upload .CSV file", type=["csv"])
    file_name_input = st.text_input("Masukkan nama file hasil preprocessing (tanpa ekstensi .csv):")
    if uploaded_file is not None:
        try :
            df = pd.read_csv(uploaded_file, dtype={"Rating":"object"}, index_col=0)
            st.dataframe(df)
        except pd.errors.EmptyDataError:
            st.write("File is empty, please check your input.")
        except pd.errors.ParserError:
            st.write("Invalid data format, please check your input.")
    
    preprocessing = st.button("Preprocessing")
    if preprocessing:
        with st.spinner('Sedang melakukan preprocessing...'):
            time.sleep(2)
            # st.success("Preprocessing Berhasil & Data Disimpan!")
            df['Ulasan'] = df['Ulasan'].fillna('')
            df['Ulasan'] = df['Ulasan'].apply(preprocessingFunction.clean)
            st.write('')
            st.write(f'--------------------------------------------------------------  CLEANING  --------------------------------------------------------------')
            st.write(df['Ulasan'])

            df['Ulasan'] = df['Ulasan'].apply(preprocessingFunction.normalisasi)
            st.write('')
            st.write(f'--------------------------------------------------------------  NORMALIZE  --------------------------------------------------------------')
            st.write(df['Ulasan'])

            df['Ulasan'] = df['Ulasan'].apply(preprocessingFunction.stopword)
            st.write('')
            st.write('--------------------------------------------------------------  STOPWORD  --------------------------------------------------------------')
            st.write(df['Ulasan'])

            df['Ulasan'] = df['Ulasan'].apply(preprocessingFunction.tokenisasi)
            st.write('')
            st.write(f'--------------------------------------------------------------  TOKENIZE  --------------------------------------------------------------')
            st.write(df['Ulasan'])

            # Melakukan stemming pada kolom "Ulasan"
            df['Ulasan'] = df['Ulasan'].apply(preprocessingFunction.stemming)
            st.write('')
            st.write('--------------------------------------------------------------  STEMMING --------------------------------------------------------------')
            st.write(df['Ulasan'])

            df['Sentimen'] = df['Rating'].apply(preprocessingFunction.labeling)
            st.write('')
            st.write(f'--------------------------------------------------------------  LABELING  --------------------------------------------------------------')
            st.write(df[['Ulasan', 'Sentimen']])
            
        # Menghentikan animasi loading
        st.spinner(False)
        df = df[['Ulasan', 'Sentimen']]
        save_path = f'data/dataHasilPreprocessing/{file_name_input}.csv'
        df.to_csv(save_path, index=False)

        # Pengkondisian Alert Crawling
        jumlah_data = len(df)
        if jumlah_data > 0:
            st.snow()
            st.success(f"Preprocessing {jumlah_data} Baris & Download Data Berhasil !")
        else:
            st.warning("Preprocessing Data Gagal")    
