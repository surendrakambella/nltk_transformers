import streamlit as st
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the VADER SentimentIntensityAnalyzer model from the pickle file
#pickle_file_path = r'sia_model.pkl'

with open('sia_model.pkl', 'rb') as file:
    sia = pickle.load(file)
 
# Define a function to analyze sentiment
def analyze_sentiment(text):
    return sia.polarity_scores(text)

# Streamlit app
st.title("Sentiment Analysis using NLTK and transformers")

# Text input
user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if user_input:
        # Analyze the sentiment of the input text
        sentiment_scores = analyze_sentiment(user_input)
        
        # Determine the sentiment with the highest probability
        highest_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        highest_score = sentiment_scores[highest_sentiment] * 100
        
        st.write(f"The sentiment with the highest probability is: **{highest_sentiment.capitalize()}** with a score of **{highest_score:.2f}%**")
    else:
        st.write("Please enter some text to analyze.")
