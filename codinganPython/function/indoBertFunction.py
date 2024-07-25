import streamlit as st
import pandas as pd
# import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from transformers import BertTokenizer, TFBertForSequenceClassification

def preprocess_data(df, use_smote=False):
    df['Sentimen'] = df['Sentimen'].map({'Positif': 1, 'Negatif': 0})
    if use_smote:
        oversample = RandomOverSampler(sampling_strategy='auto')
        X = df.drop(columns=['Sentimen'])
        y = df['Sentimen']
        X_resampled, y_resampled = oversample.fit_resample(X, y)
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
        df_resampled = df_resampled.drop_duplicates(subset=['Ulasan'])
        df_resampled = df_resampled.dropna()
        return df_resampled
    else:
        df = df.drop_duplicates(subset=['Ulasan'])
        df = df.dropna()
        return df

def tokenize_data(reviews, labels, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for review in reviews:
        encoded_dict = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)
    labels = tf.convert_to_tensor(labels)
    return input_ids, attention_masks, labels

def train_model(model, train_data, test_data, optimizer, loss, metric, epochs, batch_size):
    train_input_ids, train_attention_masks, train_labels = train_data
    test_input_ids, test_attention_masks, test_labels = test_data

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        history = model.fit(
            [train_input_ids, train_attention_masks],
            train_labels,
            batch_size=batch_size,
            epochs=1,
            validation_data=([test_input_ids, test_attention_masks], test_labels),
        )

    return history

