# Twitter Sentiment Analysis Using NLP and Machine Learning

## 📌 Project Overview

This project analyzes the sentiment of tweets using Natural Language Processing (NLP) and Machine Learning.

The system classifies tweets into:

- Positive
- Negative

The project uses the Sentiment140 dataset and a Logistic Regression model with TF-IDF feature extraction.

## 🚀 Live Demo

👉 [Open the Live App](https://twitter-sentiment-analysis-sunmz7ktdu5tbnp9h8ji4y.streamlit.app/)

## 🎯 Objective

To develop a machine learning system that can automatically determine whether the sentiment expressed in a tweet is positive or negative.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 🧠 Machine Learning

### Text Preprocessing

The tweets are cleaned by:

- Removing URLs
- Removing user mentions
- Removing hashtag symbols
- Removing special characters
- Converting text to lowercase
- Removing extra spaces

### Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical features.

### Models Compared

Three machine learning algorithms were evaluated:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 80.15% |
| Naive Bayes | 78.70% |
| Linear SVM | 78.83% |

Logistic Regression achieved the highest accuracy and was selected as the final model.

## 📊 Dataset

The project uses the Sentiment140 dataset containing 1.6 million tweets.

- 800,000 Negative tweets
- 800,000 Positive tweets

## 📈 Results

The final Logistic Regression model achieved approximately:

**80.15% Accuracy**

## 💻 Web Application

A Streamlit web application was developed where users can enter a tweet or text and receive:

- Predicted sentiment
- Prediction confidence
- Positive probability
- Negative probability

## 📁 Project Structure

Twitter_Sentiment_Analysis/

├── dataset/

├── graphs/

├── venv/

├── app.py

├── train_model.py

├── eda.py

├── model_comparison.py

├── sentiment_model.pkl

├── tfidf_vectorizer.pkl

├── requirements.txt

└── README.md

## 🚀 How to Run

### 1. Activate virtual environment

```bash
venv\Scripts\activate
