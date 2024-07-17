# Sentiment Analysis App

This Streamlit application provides sentiment analysis using two different models: a base model with TF-IDF vectorization and a zero-shot classification model using transformers.

## Features

- Text input for sentiment analysis
- Two different models for prediction:
  1. Base model using TF-IDF vectorizer
  2. Transformers zero-shot classification model
- Real-time prediction display

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Keras
- Joblib
- Pickle
- Transformers
- Custom `normalizer` module

## Installation

1. Clone this repository:
2. Install the required packages:
3. Ensure you have the following files in the correct directories:
- `basemodel/base_model.h5`
- `basemodel/tfidf_vectorizer.pkl`
- `basemodel/ohe_basemodel.joblib`

## Usage

1. Run the Streamlit app:
2. Open your web browser and go to the provided local URL (typically `http://localhost:8501`)

3. Enter text in the provided text areas for each model

4. Click the "Predict" button for the respective model to see the sentiment prediction

## Project Structure

- `app.py`: Main Streamlit application file
- `normalizer.py`: Custom module for text normalization
- `basemodel/`: Directory containing the base model and its associated files
- `requirements.txt`: List of project dependencies

## Models

1. Base Model (TF-IDF vectorizer model):
- Uses TF-IDF vectorization for text representation
- Predicts sentiment categories: Sadness, Joy, Love, Anger, Fear, Surprise

2. Transformers Model (Zero-shot classification):
- Uses the `facebook/bart-large-mnli` model for zero-shot classification
- Predicts sentiment categories: Sadness, Joy, Love, Anger, Fear, Surprise

## Contributing

Contributions are welcome. Please open an issue to discuss major changes before making a pull request.

