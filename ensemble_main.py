# ensemble_main.py

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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text.lower())
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv("dataset/tweet_emotions.csv")[['content', 'sentiment']]
df.columns = ['text', 'emotion']

# Filter top 6 emotions
top_emotions = ['happiness', 'sadness', 'neutral', 'worry', 'relief', 'love']
df = df[df['emotion'].isin(top_emotions)]

# Clean the text
df['cleaned'] = df['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['emotion'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Define models
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
rand_forest = RandomForestClassifier(n_estimators=150, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train all models
log_reg.fit(X_train, y_train)
rand_forest.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Ensemble model
voting_clf = VotingClassifier(
    estimators=[('lr', log_reg), ('rf', rand_forest), ('xgb', xgb)],
    voting='hard'
)
voting_clf.fit(X_train, y_train)

# Evaluate models
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rand_forest,
    "XGBoost": xgb,
    "Voting Ensemble": voting_clf
}

print("\n📊 MODEL ACCURACIES")
best_model = None
best_score = 0

for name, model in models.items():
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name}: {round(score * 100, 2)}%")
    if score > best_score:
        best_model = model
        best_score = score

# Save the best model and vectorizer
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Final report
print("\n✅ Selected Best Model:", best_model.__class__.__name__)
print("\n📄 Classification Report:")
print(classification_report(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(best_model.predict(X_test))
))

# Prediction function
def predict_emotion(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    encoded = best_model.predict(vector)[0]
    return label_encoder.inverse_transform([encoded])[0]

# Test prediction
sample = "I'm feeling lost, confused and very sad"
print("\n🧠 Test Prediction:")
print(f"Input: {sample}")
print("Predicted Emotion:", predict_emotion(sample))

# Bar chart of prediction distribution
df['prediction'] = label_encoder.inverse_transform(best_model.predict(X))
emotion_counts = df['prediction'].value_counts()

plt.figure(figsize=(8, 4))
emotion_counts.plot(kind='bar', color='skyblue')
plt.title("Predicted Emotion Distribution (Top 6 Emotions)")
plt.xlabel("Emotion")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("mood_plot.png")
plt.show()
