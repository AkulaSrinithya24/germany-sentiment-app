import streamlit as st
import pickle
from PIL import Image
import base64

# Set wide mode and custom page config
st.set_page_config(page_title="Germany Sentiment Classifier 🇩🇪", layout="wide")

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Custom background (optional)
def add_bg_image():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Flag_of_Germany.svg/1200px-Flag_of_Germany.svg.png');
            background-size: cover;
            background-position: top left;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

# Header content
st.markdown("<h1 style='color: #002366; font-size: 48px;'>🇩🇪 Germany Sentiment Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 20px;'>Enter a sentence related to Germany. The model will predict the sentiment as <strong style='color: green;'>Positive</strong> or <strong style='color: red;'>Negative</strong>.</p>", unsafe_allow_html=True)

# Input box
sentence = st.text_area("✍️ Type your sentence below:", height=100, placeholder="Example: Germany has a strong economy and beautiful landscapes.")

# Predict sentiment
if st.button("🔍 Analyze Sentiment"):
    if sentence.strip() != "":
        X_input = vectorizer.transform([sentence])
        prediction = model.predict(X_input)[0]

        color = "green" if prediction == "Positive" else "red"
        st.markdown(
            f"<h3 style='color:{color};'>🧠 Predicted Sentiment: <b>{prediction}</b></h3>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter a valid sentence to analyze.")

# Footer
st.markdown("---")
st.markdown("<p style='font-size: 14px; text-align: center;'>Built with ❤️ using Streamlit & Machine Learning</p>", unsafe_allow_html=True)

