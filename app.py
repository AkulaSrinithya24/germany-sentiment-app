import streamlit as st
import pickle

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Set page config
st.set_page_config(page_title="Germany Sentiment Classifier", layout="centered")

# Custom CSS for background and styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .title {
            display: flex;
            align-items: center;
            font-family: 'Segoe UI', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            color: #0713f5;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #333333;
            margin-bottom: 25px;
        }
        .input-label {
            font-size: 1rem;
            color: #6a1b9a;
            margin-bottom: 5px;
        }
        .footer {
            margin-top: 60px;
            font-size: 0.9rem;
            text-align: center;
            color: #6a1b9a;
        }
    </style>
""", unsafe_allow_html=True)

# Title with flag
st.markdown("""
<div class='title'>
    <img src='https://upload.wikimedia.org/wikipedia/en/b/ba/Flag_of_Germany.svg' width='50' style='margin-right: 12px; vertical-align: middle;'>
    Germany Sentiment Classifier
</div>
""", unsafe_allow_html=True)

# Subtitle
st.markdown("""
<div class='subtitle'>
    Enter a sentence related to Germany. The model will predict whether the sentiment is
    <b style='color:green;'>Positive</b>,
    <b style='color:orange;'>Neutral</b>, or
    <b style='color:red;'>Negative</b>.
</div>
""", unsafe_allow_html=True)

# Input
st.markdown("<div class='input-label'>📝 Type your sentence below:</div>", unsafe_allow_html=True)
sentence = st.text_input("Example: Germany has a strong economy and beautiful landscapes.")

# Prediction
if sentence:
    X_input = vectorizer.transform([sentence])
    prediction = model.predict(X_input)[0]

    if prediction.lower() == "positive":
        st.success("🧠 Predicted Sentiment: **Positive** 😊")
    elif prediction.lower() == "negative":
        st.error("🧠 Predicted Sentiment: **Negative** 😟")
    elif prediction.lower() == "neutral":
        st.info("🧠 Predicted Sentiment: **Neutral** 😐")
    else:
        st.warning(f"Prediction: {prediction}")

# Footer
st.markdown("<div class='footer'>Germany Sentiment Classifier powered by Applied Artificial Intelligence</div>", unsafe_allow_html=True)
