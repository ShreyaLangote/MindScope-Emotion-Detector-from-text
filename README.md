# 🧠 MindScope – Emotion Detection from Text

**MindScope** is an AI-powered web app that analyzes user-written text to detect emotions in real time. It enables journaling platforms, mental wellness tools, and intelligent assistants to better understand a user’s mood using simple natural language input.

This project uses a machine learning model (Logistic Regression + TF-IDF) trained on real-world tweets to classify emotions like `happiness`, `sadness`, `love`, `worry`, `relief`, and `neutral`. The final app is deployed globally using Streamlit Cloud.

---

## 🚀 Live Demo

👉 [Launch the App](https://mindscope-emotion-detector-from-text-fwxgvhuaaqdhhjhjldnm7g.streamlit.app/)

---

## ✨ Key Features

- 💬 Analyze free-form user-written text
- 🧠 Predicts 6 core emotional states
- 📉 Visualizes emotion prediction distribution
- ⚡ Lightweight ML model (<100MB) for quick deployment
- 🌐 Hosted on Streamlit Cloud (accessible worldwide)
- 📦 Easy to integrate with other systems (chatbots, wellness apps, journaling platforms)

---

## 🧠 Emotions Detected

The model is trained to recognize the following emotional states:

- `happiness`
- `sadness`
- `love`
- `worry`
- `relief`
- `neutral`

---

## 🧰 Tech Stack

| Tool / Library       | Purpose                                  |
|----------------------|-------------------------------------------|
| Python               | Programming language                     |
| Scikit-learn         | ML model (Logistic Regression)           |
| NLTK                 | Text preprocessing and lemmatization     |
| Pandas, Matplotlib   | Data analysis and visualization          |
| TF-IDF Vectorizer    | Feature extraction from text              |
| Streamlit            | Web app deployment framework             |

---

## 📊 Dataset Source

- **Title**: Emotion Detection from Text (Tweets)  
- **Source**: [Kaggle Dataset – by Pashupatigupta](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)  
- **License**: For educational and non-commercial use  
- **Preprocessing**: Cleaned, filtered to top 6 emotions, and lemmatized

---

## 📂 Project Structure

mindscope_project/
├── dataset/
│ └── tweet_emotions.csv # Training data
├── best_model.pkl # Final trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── label_encoder.pkl # Label encoder (optional use)
├── mood_plot.png # Output bar chart (emotion dist.)
├── ensemble_main.py # Model training script
├── streamlit_app.py # Streamlit frontend UI
├── requirements.txt # Python dependencies
└── README.md # Project overview

yaml
Copy code

---

## 🧪 Run Locally


git clone https://github.com/ShreyaLangote/MindScope-Emotion-Detector-from-text.git
cd MindScope-Emotion-Detector-from-text

# Install required libraries
pip install -r requirements.txt

# Start the app
streamlit run streamlit_app.py
👩‍💻 Author
Shreya Langote
🎓 AI/ML • B.Tech (Computer Science)
🔗 LinkedIn-linkedin.com/in/shreya-langote-7729702b5
💻 GitHub-ShreyaLangote

🙌 Acknowledgments
Dataset Creator: @pashupatigupta on Kaggle

Streamlit for open-source web app hosting

Scikit-learn and NLTK teams for robust ML tools

🤝 Contributing
Have ideas to improve accuracy or add deep learning? Contributions are welcome!

bash
Copy code
# Fork this repo
# Make your changes
# Submit a pull request 🚀
📄 License
This project is licensed under the MIT License — feel free to use, modify, and distribute.
