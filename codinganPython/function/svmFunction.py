from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st

# def select_model(model_name):
#     if model_name == 'Support Vector Machine':
#         return SVC()
#     elif model_name == 'DecisionTreeClassifier':
#         return DecisionTreeClassifier()
#     elif model_name == 'MultinomialNB':
#         return MultinomialNB()
#     else:
#         return None

# Pastikan direktori 'model' ada
if not os.path.exists('model'):
    os.makedirs('model')

def analyze_sentiment(data, model_name, test_size, model_filename):
    try:
        # Menangani NaN pada kolom teks
        data['Ulasan'] = data['Ulasan'].fillna("")

        train_data, test_data, train_labels, test_labels = train_test_split(
            data['Ulasan'], data['Sentimen'], test_size=test_size, random_state=42
        )

        vectorizer = CountVectorizer()
        train_features = vectorizer.fit_transform(train_data)
        test_features = vectorizer.transform(test_data)

        model = model_name
        model.fit(train_features, train_labels)

        model_filename_with_extension = f'codinganPython/model/{model_filename}.pkl'
        joblib.dump(model, model_filename_with_extension)

        vectorizer_filename = f'codinganPython/model/{model_filename}_vectorizer.pkl'
        joblib.dump(vectorizer, vectorizer_filename)

        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions)
        unique_labels = sorted(data['Sentimen'].unique())
        cm = confusion_matrix(test_labels, predictions, labels=unique_labels)

        fig = ff.create_annotated_heatmap(
            z=cm,
            x=unique_labels,
            y=unique_labels,
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(
            xaxis=dict(title='Predicted'),
            yaxis=dict(title='True'),
            title='Confusion Matrix',
        )

        return accuracy, report, model_filename_with_extension, vectorizer_filename, fig

    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None, None

def predict_sentiment(model, vectorizer, text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return prediction

# Fungsi untuk menghitung jumlah kata berdasarkan segmentasi dan sentimen
def get_segmented_word_counts(data, segmentation, sentiment_col='Sentimen'):
    segmented_word_counts = {seg: {'Positif': 0, 'Negatif': 0} for seg in segmentation.keys()}
    
    for _, row in data.iterrows():
        sentiment = row.get(sentiment_col, None)
        processed_text = row.get('processed', [])
        
        if sentiment is None:
            print(f"Sentimen tidak ditemukan di baris: {row}")
            continue
        if not isinstance(processed_text, list):
            print(f"Kolom 'processed' tidak berisi list di baris: {row}")
            continue

        for word in processed_text:
            segment = categorize_word(word, segmentation)
            if segment:
                segmented_word_counts[segment][sentiment] += 1
    
    return segmented_word_counts

def categorize_word(word, segmentation):
    for segment, keywords in segmentation.items():
        if word in keywords:
            return segment
    return None

def plot_segmented_word_counts(segmented_word_counts):
    fig, ax = plt.subplots(figsize=(12, 8))
    df_list = []
    for segment, counts in segmented_word_counts.items():
        for sentiment, count in counts.items():
            df_list.append({'Segmentasi': segment, 'Sentimen': sentiment, 'JumlahKata': count})
    
    df = pd.DataFrame(df_list)
    sns.barplot(data=df, x='Segmentasi', y='JumlahKata', hue='Sentimen', ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.title('Jumlah Kata Berdasarkan Segmentasi dan Sentimen')
    return fig