import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Germany Sentiment App 🇩🇪",
    page_icon="🇩🇪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS styling
st.markdown("""
    <style>
    body {
        background-color: #0d1117;
        color: white;
    }
    .main {
        background-color: #161b22;
        padding: 2rem;
        border-radius: 12px;
    }
    .stTextInput>div>div>input {
        background-color: #0d1117;
        color: white;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    .title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #c9d1d9;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        font-size: 0.9rem;
        color: #8b949e;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🇩🇪 Germany Sentiment Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🔍 Enter a sentence about Germany and let our machine learning model predict whether the sentiment is <b>Positive</b> or <b>Negative</b>.</div>', unsafe_allow_html=True)

# Input
sentence = st.text_input("📝 Enter a sentence:", "")

# Prediction
if sentence:
    vec = vectorizer.transform([sentence])
    prediction = model.predict(vec)[0]

    # Display result
    if prediction == "Positive":
        st.success("🧠 Predicted Sentiment: **Positive** 😊")
    else:
        st.error("🧠 Predicted Sentiment: **Negative** 😞")

st.markdown('<div class="footer">Made with ❤️ using Streamlit | Applied AI Project</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
