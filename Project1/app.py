import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (only first run)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load vectorizer and model
cv = pickle.load(open('./cv.pkl', 'rb'))
model = pickle.load(open('./model.pkl', 'rb'))

# Initialize Lemmatizer
lemma = WordNetLemmatizer()

# Title
st.title("🎬 Movie Review Sentiment Analysis")

# User input
input_review = st.text_input("Enter the review")

if st.button("Analyze"):
    if input_review.strip() != "":
        new_message = pd.DataFrame({'message': [input_review]})

        new_corpus = []
        for i in range(len(new_message)):
            review = re.sub('[^a-zA-Z]', ' ', new_message['message'][i])
            review = review.lower()
            review = review.split()
            review = [lemma.lemmatize(word) for word in review if word not in stopwords.words('english')]
            review = ' '.join(review)
            new_corpus.append(review)

        X_new = cv.transform(new_corpus).toarray()
        y_pred = model.predict(X_new)[0]

        if y_pred == 1:
            st.header("✅ Positive Review")
        else:
            st.header("❌ Negative Review")
    else:
        st.warning("Please enter a review.")
