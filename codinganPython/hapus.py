import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.figure_factory as ff
import streamlit as st

def select_model(model_name):
    if model_name == 'RandomForestClassifier':
        return RandomForestClassifier()
    elif model_name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier()
    elif model_name == 'MultinomialNB':
        return MultinomialNB()
    elif model_name == 'Support Vector Machine':
        from sklearn.svm import SVC
        return SVC()
    else:
        return None

def analyze_sentiment(data, model_name, test_size, model_filename):
    try:
        # Menangani NaN pada kolom teks
        data['Ulasan'] = data['Ulasan'].fillna("")

        # Membagi data menjadi data latih dan data uji
        train_data, test_data, train_labels, test_labels = train_test_split(
            data['Ulasan'], data['Sentimen'], test_size=test_size, random_state=42
        )

        # Membuat dan melatih CountVectorizer
        vectorizer = CountVectorizer()
        train_features = vectorizer.fit_transform(train_data)
        test_features = vectorizer.transform(test_data)

        # Memilih model berdasarkan nama model yang diberikan
        model = select_model(model_name)

        # Memastikan model yang dipilih valid
        if model is None:
            raise ValueError(f"Model {model_name} tidak didukung")

        # Melatih model
        model.fit(train_features, train_labels)

        # Menyimpan model dan vectorizer
        model_filename_with_extension = f'{model_filename}.pkl'
        joblib.dump(model, model_filename_with_extension)

        vectorizer_filename = f'{model_filename}_vectorizer.pkl'
        joblib.dump(vectorizer, vectorizer_filename)

        # Melakukan prediksi dan menghitung akurasi
        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions)
        unique_labels = sorted(data['Sentimen'].unique())
        cm = confusion_matrix(test_labels, predictions, labels=unique_labels)

        # Membuat heatmap dari confusion matrix
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

        # Mengembalikan hasil yang diinginkan
        return accuracy, report, model_filename_with_extension, vectorizer_filename, fig

    except Exception as e:
        # Mengeluarkan pesan kesalahan
        st.error(f"Terjadi kesalahan: {str(e)}")
        return None, None, None, None, None

uploaded_file = st.file_uploader("Upload Excel file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        model_name = st.selectbox("Select Model", ["RandomForestClassifier", "DecisionTreeClassifier", "MultinomialNB", "Support Vector Machine"])
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
        model_filename = st.text_input("Input Model Filename (without extension):")

        if model_filename and st.button("Start Analysis"):
            accuracy, report, model_filename_with_extension, vectorizer_filename, fig = analyze_sentiment(data, model_name, test_size, model_filename)

            if accuracy is not None and report is not None:
                st.write(f"Accuracy: {accuracy:.2f}")
                st.text("Classification Report:")
                st.text(report)
                st.plotly_chart(fig)
                st.success(f"Model saved to {model_filename_with_extension}")
                st.success(f"Vectorizer saved to {vectorizer_filename}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
