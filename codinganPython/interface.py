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
from function import indoBertFunction
import os
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler

with st.sidebar:
    selected = option_menu("Main Menu", ["Dashboard", "Crawling", "Merge Data", 'Dataset', 'Preprocessing', "Visualization", "Support Vector Machine", "IndoBert", "Testing"], 
        icons=['house', 'gear', 'book', 'pen', 'pen', 'book', 'kanban','activity', 'activity', 'cloud-upload' ], menu_icon="cast", default_index=0)
    selected

if selected == 'Dashboard':
    st.title("Dashboard :")
    st.subheader("Data")
    df_dashboard = pd.read_csv("data/dataHasilPenggabungan/dataSentimenProduk1-10.csv")
    df_dashboard = df_dashboard[['Nama Pelanggan', 'Produk', 'Ulasan', 'Rating']]
    dfVisualization = pd.read_csv("data/dataHasilPreprocessing/hasilPreprocessing1.csv")
    if 'Ulasan' not in df_dashboard.columns:
        st.warning("Data yang dimasukkan tidak sesuai.")
    else:
        st.dataframe(df_dashboard)

    with st.spinner('Performing Visualization...'):
        if 'Sentimen' not in dfVisualization.columns:
            st.warning("Data yang dimasukkan tidak sesuai.")
        else:
            st.subheader("Visualization Sentiment - Bar Chart :")
            custom_palette = {'Negatif': 'red', 'Positif': '#0384fc'}
            plt.figure(figsize=(10, 6))

            ax = sns.countplot(x='Sentimen', data=dfVisualization, order=['Negatif', 'Positif'], palette=custom_palette)
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

# Alur program di Streamlit
elif selected == "Crawling":
    col1, col2 = st.columns([1,8])
    with col1:
        st.image('img/tokopedia.png', width=80)
    with col2:
        st.title("Crawling Data")

    url = st.text_input("Masukkan URL produk:")
    nama_file = st.text_input("Masukkan nama file hasil scraping data:")
    jumlah_data = st.number_input("Masukkan jumlah data yang ingin diambil:", min_value=1, step=1, value=10)
    rating_min = st.number_input("Masukkan rating minimum yang ingin diambil (1-5):", min_value=1, max_value=5, step=1, value=1)
    rating_max = st.number_input("Masukkan rating maksimum yang ingin diambil (1-5):", min_value=1, max_value=5, step=1, value=5)
    folder_path = "data/dataScrapingHanaShop/"
    file_path = folder_path + nama_file

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

elif selected == "Dataset":
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
                            custom_palette = {'Negatif': 'red', 'Positif': '#0384fc'}
                            plt.figure(figsize=(10, 6))

                            ax = sns.countplot(x='Sentimen', data=dfVisualization, order=['Negatif', 'Positif'], palette=custom_palette)
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
            model_name = SVC()
            test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
            model_filename = st.text_input("Input Model Filename (without extension):")
            smote_option = st.selectbox("SMOTE Option", ["SMOTE", "TANPA SMOTE"])

            if model_filename and st.button("Start Analysis"):
                if smote_option == "SMOTE":
                    # Proses SMOTE
                    data_resampled = data.copy()
                    X = data_resampled.drop(columns=['Sentimen'])
                    y = data_resampled['Sentimen']
                    oversample = RandomOverSampler(sampling_strategy='auto')
                    X_resampled, y_resampled = oversample.fit_resample(X, y)
                    data_resampled = pd.concat([X_resampled, y_resampled], axis=1)

                    # Visualisasi jumlah sentimen sebelum SMOTE
                    sentimen_before = data['Sentimen'].value_counts()
                    st.write("Before SMOTE:")
                    st.bar_chart(sentimen_before)

                    # Visualisasi jumlah sentimen sesudah SMOTE
                    sentimen_after = data_resampled['Sentimen'].value_counts()
                    st.write("After SMOTE:")
                    st.bar_chart(sentimen_after)

                accuracy, report, model_filename_with_extension, vectorizer_filename, fig = svmFunction.analyze_sentiment(data_resampled if smote_option == "SMOTE" else data, model_name, test_size, model_filename)

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
    uploaded_file = st.file_uploader("Upload csv file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        use_smote = st.checkbox("Use SMOTE")

        # if use_smote:
        #     st.write("Data before SMOTE:")
        #     sentiment_counts_before = df['Sentimen'].value_counts()
        #     st.bar_chart(sentiment_counts_before)

        df = indoBertFunction.preprocess_data(df, use_smote)

        model_name = 'indobenchmark/indobert-base-p1'
        tokenizer = indoBertFunction.BertTokenizer.from_pretrained(model_name)
        model = indoBertFunction.TFBertForSequenceClassification.from_pretrained(model_name)

        reviews = df['Ulasan'].tolist()
        labels = df['Sentimen'].tolist()

        max_length = 128
        input_ids, attention_masks, labels = indoBertFunction.tokenize_data(reviews, labels, tokenizer, max_length)

        train_indices, test_indices = train_test_split(range(len(input_ids)), test_size=0.2, random_state=42)
        train_data = (tf.gather(input_ids, train_indices), tf.gather(attention_masks, train_indices), tf.gather(labels, train_indices))
        test_data = (tf.gather(input_ids, test_indices), tf.gather(attention_masks, test_indices), tf.gather(labels, test_indices))

        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        epochs = st.number_input("Masukkan Jumlah Epoch", min_value=1, max_value=20, value=10, step=1)
        batch_size = 16

        if st.button("Start Training"):
            st.write("Training in progress...")
            history = indoBertFunction.train_model(model, train_data, test_data, optimizer, loss, metric, epochs, batch_size)
            st.write("Training completed!")

            st.write("Evaluation:")
            model.evaluate([test_data[0], test_data[1]], test_data[2])

            test_predictions = model.predict([test_data[0], test_data[1]])
            predicted_labels = tf.argmax(test_predictions.logits, axis=1)

            st.write("Classification Report:")
            st.text(classification_report(test_data[2], predicted_labels))

elif selected == 'Testing':
    st.title("Testing :")
    model_file = st.file_uploader('Pilih file model (pkl)', type=['pkl'])
    vectorizer_file = st.file_uploader('Pilih file vectorizer (pkl)', type=['pkl'])

    if model_file and vectorizer_file:
        # Load model dan vectorizer dari file yang diunggah
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
        
        # Input teks dari pengguna
        user_input = st.text_area('Masukkan teks untuk diterjemahkan dan dianalisis:')
        
        # Jika tombol ditekan untuk menganalisis
        if st.button('Terjemahkan dan Prediksi'):
            if user_input:
                # Panggil fungsi predict_sentiment dengan model, vectorizer, dan teks sebagai argumen
                sentiment = svmFunction.predict_sentiment(model, vectorizer, user_input)
                
                # Tampilkan hasil prediksi sentimen
                st.write('Sentimen:', sentiment)
            else:
                st.warning('Masukkan teks untuk menganalisis.')
    else:
        st.warning('Pilih file model dan vectorizer sebelum melakukan testing.')