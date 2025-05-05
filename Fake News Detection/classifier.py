# Import necessary libraries
import streamlit as st
import pandas as pd
import re
import string
import joblib
from collections import Counter

# Load pre-trained models and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
SVM = joblib.load("SVM_model.pkl")  
DT = joblib.load("DT_model.pkl")
GBC = joblib.load("GBC_model.pkl")
RFC = joblib.load("RFC_model.pkl")

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Label output
def output_label(n):
    return "🟥 Fake News" if n == 0 else "🟩 Real News"

# Prediction using majority voting
def manual_testing(news):
    df_test = pd.DataFrame({"text": [news]})
    df_test["text"] = df_test["text"].apply(wordopt)
    x_test_vec = vectorizer.transform(df_test["text"])

    preds = [
        SVM.predict(x_test_vec)[0],  
        DT.predict(x_test_vec)[0],
        GBC.predict(x_test_vec)[0],
        RFC.predict(x_test_vec)[0]
    ]

    majority_vote = Counter(preds).most_common(1)[0][0]
    return output_label(majority_vote)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")

news_input = st.text_area("Enter news text below:")

if st.button("Check"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        prediction = manual_testing(news_input)
        st.subheader("🧠 Final Prediction Based on Majority Vote")
        st.success(prediction)
