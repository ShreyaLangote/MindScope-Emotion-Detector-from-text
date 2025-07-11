# streamlit_app.py

import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model components
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text.lower())
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Emotion to emoji map
emoji_map = {
    "happiness": "😊",
    "sadness": "😢",
    "love": "❤️",
    "worry": "😟",
    "relief": "😌",
    "neutral": "😐"
}

# Streamlit UI
st.set_page_config(page_title="MindScope - Emotion Detector", page_icon="🧠")
st.title("MindScope - Emotion Detection from Your Text")

st.markdown("""
Type how you're feeling, what you're thinking, or even your journal entry.  
Our AI model will analyze your words and tell you what emotion you're expressing.
""")

user_input = st.text_area("✍️ Write here:", height=150)

if st.button("🔍 Detect Emotion"):
    if not user_input.strip():
        st.warning("Please enter something first.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        encoded = model.predict(vector)[0]
        emotion = label_encoder.inverse_transform([encoded])[0]
        emoji = emoji_map.get(emotion, "❓")
        
        st.success(f"**Predicted Emotion:** {emotion.upper()} {emoji}")

        # Optional: Show feedback or suggestion
        if emotion == "sadness":
            st.info("🌈 Try listening to your favorite song or going for a walk.")
        elif emotion == "worry":
            st.info("🫶 Remember to breathe deeply — everything will be okay.")

# Toggle for chart
if st.checkbox("📊 Show Mood Distribution Chart"):
    st.image("mood_plot.png", caption="Overall Emotion Prediction Distribution")
    
st.markdown("---")
st.markdown("Created with ❤️ by Shreya | Powered by Ensemble ML")
