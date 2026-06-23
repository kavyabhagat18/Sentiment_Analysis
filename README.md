# Airline Tweet Sentiment Classifier

Classifies feedback text as **Positive**, **Neutral**, or **Negative**, trained on the
Twitter US Airline Sentiment dataset.

## Project structure

sentiment_app/
├── data/Tweets.csv                  # dataset
├── models/sentiment_model.pkl       # trained Logistic Regression model
├── models/tfidf_vectorizer.pkl      # fitted TF-IDF vectorizer
├── outputs/evaluation_metrics.json  # accuracy / precision / recall / F1
├── outputs/confusion_matrix.png     # confusion matrix plot
├── preprocessing.py                 # NLTK text-cleaning pipeline
├── train_model.py                   # training + evaluation script
├── app.py                           # Streamlit web app
└── requirements.txt

## Setup

pip install -r requirements.txt

## Train the model

python3 train_model.py

## Run the web app

streamlit run app.py
