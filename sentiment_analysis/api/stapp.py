import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
import pickle
from normalizer import normalize_text
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')

# Load the base model
base_model = load_model('basemodel/base_model.h5')

# load the vectorizer TfidfVectorizer 
with open('basemodel/tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
    
# load the encoder joblib
with open('basemodel/ohe_basemodel.joblib', 'rb') as file:
    encoder_basemodel = joblib.load(file)

# Load the transformers zero-shot classification model
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')



# Function to preprocess and predict text using the base model
def predict_text_basemodel(text: str):
    # normalize the text
    text = normalize_text(text)
    # convert the text to a list
    text = [text]
    # vectorize the text
    text_vector = vectorizer.transform(text)
    # turn vector into array
    text_vector = text_vector.toarray()
    # predict the text
    prediction = base_model.predict(text_vector)
    # encode the prediction
    prediction = encoder_basemodel.inverse_transform(prediction)
    return prediction

# Function to predict text using the transformers zero-shot classification model
def predict_text_transformers(text: str):
    prediction = classifier(text, candidate_labels=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'])
    return prediction['labels'][0]

# Streamlit application structure
st.title('Sentiment Analysis App')


# Model 1: Base Model
st.header('Model 1: TF-IDF vectorizer model')
text_input_2 = st.text_area("Enter text:", key='text_input_2')
if st.button('Predict Model 1', key='predict_button_2'):
    if text_input_2:
        prediction_2 = predict_text_basemodel(text_input_2)
        st.success(f"Prediction: {prediction_2}")

# Model 2: Transformers
st.header('Model 2: Transformers, Zero-shot classification model')
text_input_3 = st.text_area("Enter text:", key='text_input_3')
if st.button('Predict Model 2', key='predict_button_3'):
    if text_input_3:
        prediction_3 = predict_text_transformers(text_input_3)
        st.success(f"Prediction: {prediction_3}")








