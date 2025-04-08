import streamlit as st
import pickle

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Streamlit UI
st.title("🇩🇪 Germany Sentiment Classifier")
st.markdown("Enter a sentence related to Germany. The model will predict whether the sentiment is **Positive** or **Negative**.")

# Input box
sentence = st.text_input("Enter a sentence:")

# Prediction
if sentence:
    X_input = vectorizer.transform([sentence])
    prediction = model.predict(X_input)[0]
    st.success(f"🧠 Predicted Sentiment: **{prediction}**")
