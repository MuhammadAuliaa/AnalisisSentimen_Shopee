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

def select_model(model_name):
    if model_name == 'RandomForestClassifier':
        return RandomForestClassifier()
    elif model_name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier()
    elif model_name == 'MultinomialNB':
        return MultinomialNB()
    else:
        return None

def analyze_sentiment(data, model_name, test_size, model_filename):
    try:
        train_data, test_data, train_labels, test_labels = train_test_split(
            data['Ulasan'], data['Sentimen'], test_size=test_size, random_state=42
        )

        vectorizer = CountVectorizer()
        train_features = vectorizer.fit_transform(train_data)
        test_features = vectorizer.transform(test_data)
        model = select_model(model_name)
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
        return None, None, None, None, None

def predict_sentiment(model, vectorizer, text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return prediction