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
import scrapingFunction

selected = option_menu(None, ["Crawling", "Dataset"], 
icons=['cloud-upload', "archive", 'gear', 'activity', 'kanban', 'kanban'], 
menu_icon="cast", default_index=0, orientation="horizontal")

if selected == 'Crawling':
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

if selected == 'Dataset':
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

