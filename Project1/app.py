import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import numpy as np
import os 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (only first run)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load vectorizer and model.
base_dir = os.path.dirname(__file__)
with open(os.path.join(base_dir, 'cv.pkl'), 'rb') as f:
    cv = pickle.load(f)
    
with open(os.path.join(base_dir, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

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
