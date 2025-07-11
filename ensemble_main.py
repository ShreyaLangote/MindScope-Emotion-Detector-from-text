import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text.lower())
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Load data
df = pd.read_csv("dataset/tweet_emotions.csv")[['content', 'sentiment']]
df.columns = ['text', 'emotion']
top_emotions = ['happiness', 'sadness', 'neutral', 'worry', 'relief', 'love']
df = df[df['emotion'].isin(top_emotions)]
df['cleaned'] = df['text'].apply(clean_text)

# TF-IDF with fewer features
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['cleaned'])
y = df['emotion']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Encode for saving
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)

# Model: Use only logistic regression (smallest)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Evaluate
y_pred = log_reg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy: {acc * 100:.2f}%")

# Save lightweight models
with open("best_model.pkl", "wb") as f:
    pickle.dump(log_reg, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Check file size
model_size = os.path.getsize("best_model.pkl") / (1024 * 1024)
print(f"📦 Model size: {model_size:.2f} MB")

# Sample prediction
def predict_emotion(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return log_reg.predict(vec)[0]

sample = "I'm feeling stuck and overwhelmed"
print("\n🧠 Test Prediction:")
print(f"Input: {sample}")
print("Predicted Emotion:", predict_emotion(sample))

# Save chart
df['prediction'] = log_reg.predict(X)
emotion_counts = df['prediction'].value_counts()
plt.figure(figsize=(8, 4))
emotion_counts.plot(kind='bar', color='skyblue')
plt.title("Predicted Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("mood_plot.png")
plt.show()
