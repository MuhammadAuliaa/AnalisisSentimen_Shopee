from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
import joblib

# def select_model(model_name):
#     if model_name == 'Support Vector Machine':
#         return SVC()
#     elif model_name == 'DecisionTreeClassifier':
#         return DecisionTreeClassifier()
#     elif model_name == 'MultinomialNB':
#         return MultinomialNB()
#     else:
#         return None

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
        model = SVC()
        model.fit(train_features, train_labels)
        model_filename_with_extension = f'./codinganPython/model/{model_filename}.pkl'

        joblib.dump(model, model_filename_with_extension)

        vectorizer_filename = f'./codinganPython/model/{model_filename}_vectorizer.pkl'
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
        return None, None, None, None, None

def predict_sentiment(model, vectorizer, text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return prediction

# def analyze_sentiment(data, model_name, test_size, model_filename, num_topics):
#     try:
#         # Menangani NaN pada kolom teks
#         data['Ulasan'] = data['Ulasan'].fillna("")

#         # Membuat kamus (dictionary) dari token
#         tokenized_reviews = [review.lower().split() for review in data['Ulasan']]
#         dictionary = corpora.Dictionary(tokenized_reviews)

#         # Membuat corpus (bag of words) dari token
#         corpus = [dictionary.doc2bow(review) for review in tokenized_reviews]

#         # Melatih model HDP untuk menemukan topik/aspek
#         hdp_model = HdpModel(corpus, dictionary, T=num_topics)

#         # Mendapatkan distribusi topik untuk setiap ulasan
#         topic_distributions = [hdp_model[doc] for doc in corpus]

#         # Mendapatkan bobot topik untuk setiap ulasan
#         X = []
#         for dist in topic_distributions:
#             feat = [0] * num_topics
#             for topic, weight in dist:
#                 if topic < num_topics:
#                     feat[topic] = weight
#             X.append(feat)

#         train_data, test_data, train_labels, test_labels = train_test_split(
#             X, data['Sentimen'], test_size=test_size, random_state=42
#         )

#         model = SVC()
#         model.fit(train_data, train_labels)
#         model_filename_with_extension = f'./codinganPython/model/{model_filename}.pkl'

#         joblib.dump(model, model_filename_with_extension)

#         predictions = model.predict(test_data)
#         accuracy = accuracy_score(test_labels, predictions)
#         report = classification_report(test_labels, predictions)
#         unique_labels = sorted(data['Sentimen'].unique())
#         cm = confusion_matrix(test_labels, predictions, labels=unique_labels)

#         fig = ff.create_annotated_heatmap(
#             z=cm,
#             x=unique_labels,
#             y=unique_labels,
#             colorscale='Blues',  
#             showscale=True
#         )
#         fig.update_layout(
#             xaxis=dict(title='Predicted'),
#             yaxis=dict(title='True'),
#             title='Confusion Matrix',
#         )

#         return accuracy, report, model_filename_with_extension, fig

#     except Exception as e:
#         return None, None, None, None

# interface
#  uploaded_file = st.file_uploader("Upload Excel file", type=["csv"])

#     if uploaded_file is not None:
#         try:
#             data = pd.read_csv(uploaded_file)
#             model_name = SVC()
#             test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
#             model_filename = st.text_input("Input Model Filename (without extension):")
#             smote_option = st.selectbox("SMOTE Option", ["SMOTE", "TANPA SMOTE"])
#             num_topics = st.number_input("Input Number of Topics", min_value=1, max_value=10, value=5)

#             if model_filename and st.button("Start Analysis"):
#                 if smote_option == "SMOTE":
#                     # Proses SMOTE
#                     data_resampled = data.copy()
#                     X = data_resampled.drop(columns=['Sentimen'])
#                     y = data_resampled['Sentimen']
#                     oversample = RandomOverSampler(sampling_strategy='auto')
#                     X_resampled, y_resampled = oversample.fit_resample(X, y)
#                     data_resampled = pd.concat([X_resampled, y_resampled], axis=1)

#                     # Visualisasi jumlah sentimen sebelum SMOTE
#                     sentimen_before = data['Sentimen'].value_counts()
#                     st.write("Before SMOTE:")
#                     st.bar_chart(sentimen_before)

#                     # Visualisasi jumlah sentimen sesudah SMOTE
#                     sentimen_after = data_resampled['Sentimen'].value_counts()
#                     st.write("After SMOTE:")
#                     st.bar_chart(sentimen_after)

#                 accuracy, report, model_filename_with_extension, fig = svmFunction.analyze_sentiment(data_resampled if smote_option == "SMOTE" else data, model_name, test_size, model_filename, num_topics)

#                 if accuracy is not None and report is not None:
#                     st.write(f"Accuracy: {accuracy:.2f}")
#                     st.text("Classification Report:")
#                     st.text(report)
#                     st.plotly_chart(fig)
#                     st.success(f"Model saved to {model_filename_with_extension}")

#         except Exception as e:
#             st.warning("Data tidak sesuai. Pastikan file yang diunggah memiliki format yang benar dan kolom yang diperlukan.")