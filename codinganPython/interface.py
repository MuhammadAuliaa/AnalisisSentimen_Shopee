import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
import seaborn as sns
from streamlit_option_menu import option_menu
import pandas as pd
import time
from function import scrapingFunction
from function import preprocessingFunction
from function import mergedataFunction
from function import svmFunction
import os
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud


with st.sidebar:
    selected = option_menu("Main Menu", ["Crawling", "Merge Data", 'Dataset', 'Preprocessing', "Visualization", "Support Vector Machine", "IndoBert", "Testing"], 
        icons=['house', 'gear', 'book', 'pen', 'pen', 'book', 'kanban','activity', 'activity', 'cloud-upload' ], menu_icon="cast", default_index=0)
    selected

# Alur program di Streamlit
if selected == "Crawling":
    col1, col2 = st.columns([1,8])
    with col1:
        st.image('img/tokopedia.png', width=80)
    with col2:
        st.title("Crawling Data")

    # Input URL produk
    url = st.text_input("Masukkan URL produk:")

    # Input nama file hasil scraping data
    nama_file = st.text_input("Masukkan nama file hasil scraping data:")

    # Input jumlah data yang ingin diambil
    jumlah_data = st.number_input("Masukkan jumlah data yang ingin diambil:", min_value=1, step=1, value=10)
    
    # Input rentang rating yang ingin diambil
    rating_min = st.number_input("Masukkan rating minimum yang ingin diambil (1-5):", min_value=1, max_value=5, step=1, value=1)
    rating_max = st.number_input("Masukkan rating maksimum yang ingin diambil (1-5):", min_value=1, max_value=5, step=1, value=5)
    
    # Tentukan folder path dan file path
    folder_path = "data/dataScrapingHanaShop/"
    file_path = folder_path + nama_file

    # Tombol untuk mulai scraping
    if st.button("Mulai Scraping"):
        scrapingFunction.scrape_tokopedia_reviews(url, jumlah_data, file_path, rating_min, rating_max)
        st.success(f"Data telah disimpan ke: {file_path}.csv")

elif selected == 'Merge Data':
    st.title("Merge Data")
    uploaded_files = st.file_uploader("Gabungkan File (*minimal 2 file)", type="csv", accept_multiple_files=True)
    merged_file_name = st.text_input("Masukkan Nama File Hasil Penggabungan (tanpa ekstensi)", "merged_data")

    if uploaded_files:
        if len(uploaded_files) < 2:
            st.warning("Mohon unggah minimal 2 file untuk melakukan penggabungan data.")
        else:
            dataframes = [pd.read_csv(file) for file in uploaded_files]
            merged_data = mergedataFunction.merge_and_reset_index(dataframes)
            st.snow()

            st.write("Merged Data:")
            st.dataframe(merged_data)

            if st.button("Download Data Hasil Penggabungan"):
                output_folder = "data/dataHasilPenggabungan"
                os.makedirs(output_folder, exist_ok=True)
                output_file_path = os.path.join(output_folder, f"{merged_file_name}.csv")
                merged_data.to_csv(output_file_path, index=False)

                st.success(f"Data penggabungan berhasil diunduh.")

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

elif selected == 'Visualization':
    st.title("Visualization:")
    uploaded_file = st.file_uploader("Upload CSV file (Max 100 Baris Data)", type=["csv"])

    if uploaded_file is not None:
        try:
            dfVisualization = pd.read_csv(uploaded_file)

            if 'Ulasan' not in dfVisualization.columns:
                st.warning("Data yang dimasukkan tidak sesuai.")
            else:
                st.dataframe(dfVisualization)

                Visualization = st.button("Visualization")

                if Visualization:
                    with st.spinner('Performing Visualization...'):
                        if 'Sentimen' not in dfVisualization.columns:
                            st.warning("Data yang dimasukkan tidak sesuai.")
                        else:
                            st.subheader("Visualization Sentiment - Bar Chart :")
                            custom_palette = {'Negatif': 'red', 'Positif': '#0384fc', 'Netral': '#787878'}
                            plt.figure(figsize=(10, 6))

                            ax = sns.countplot(x='Sentimen', data=dfVisualization, order=['Negatif', 'Positif', 'Netral'], palette=custom_palette)
                            ax.grid(axis='y', linestyle='--', alpha=0.5)
                            plt.title('Distribution of Sentiment Attributes')
                            plt.xlabel('Sentiment Attribute')
                            plt.ylabel('Count')
                            for p in ax.patches:
                                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                            st.pyplot(plt)

                            # Check if data for each sentiment exists
                            positive_messages = dfVisualization[dfVisualization['Sentimen'] == 'Positif']['Ulasan']
                            if positive_messages.empty:
                                st.warning("Data sentimen positif tidak ditemukan.")
                            else:
                                st.subheader("Visualization Text - WordCloud Positif :")
                                positive_text = ' '.join(positive_messages.astype(str))
                                wordcloud_positive = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(positive_text)
                                plt.figure(figsize=(10, 5))
                                plt.imshow(wordcloud_positive, interpolation='bilinear')
                                plt.axis('off')
                                st.pyplot(plt)

                            neutral_messages = dfVisualization[dfVisualization['Sentimen'] == 'Netral']['Ulasan']
                            if neutral_messages.empty:
                                st.warning("Data sentimen netral tidak ditemukan.")
                            else:
                                st.subheader("Visualization Text - WordCloud Netral :")
                                neutral_text = ' '.join(neutral_messages.astype(str))
                                wordcloud_neutral = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(neutral_text)
                                plt.figure(figsize=(10, 5))
                                plt.imshow(wordcloud_neutral, interpolation='bilinear')
                                plt.axis('off')
                                st.pyplot(plt)

                            negative_messages = dfVisualization[dfVisualization['Sentimen'] == 'Negatif']['Ulasan']
                            if negative_messages.empty:
                                st.warning("Data sentimen negatif tidak ditemukan.")
                            else:
                                st.subheader("Visualization Text - WordCloud Negatif :")
                                negative_text = ' '.join(negative_messages.astype(str))
                                wordcloud_negative = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)
                                plt.figure(figsize=(10, 5))
                                plt.imshow(wordcloud_negative, interpolation='bilinear')
                                plt.axis('off')
                                st.pyplot(plt)

                    st.spinner(False)

        except pd.errors.EmptyDataError:
            st.warning("Uploaded file is empty. Please upload a valid CSV file.")

elif selected == 'Support Vector Machine':
    st.title("Training Model :")
    uploaded_file = st.file_uploader("Upload Excel file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            model_name = st.selectbox("Select Model", ["RandomForestClassifier", "DecisionTreeClassifier", "MultinomialNB"])
            test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
            model_filename = st.text_input("Input Model Filename (without extension):")

            if model_filename and st.button("Start Analysis"):
                accuracy, report, model_filename_with_extension, vectorizer_filename, fig = svmFunction.analyze_sentiment(data, model_name, test_size, model_filename)

                if accuracy is not None and report is not None:
                    st.write(f"Accuracy: {accuracy:.2f}")
                    st.text("Classification Report:")
                    st.text(report)
                    st.plotly_chart(fig)
                    st.success(f"Model saved to {model_filename_with_extension}")
                    st.success(f"Vectorizer saved to {vectorizer_filename}")

        except Exception as e:
            st.warning("Data tidak sesuai. Pastikan file yang diunggah memiliki format yang benar dan kolom yang diperlukan.")

elif selected == 'IndoBert':
    st.title("Training IndoBert :")

elif selected == 'Testing':
    st.title("Testing :")