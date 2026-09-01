import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib


# ==============================
# 1. LOAD DATASET
# ==============================

file_path = "dataset/training.1600000.processed.noemoticon.csv"

columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]

df = pd.read_csv(
    file_path,
    encoding="latin-1",
    names=columns
)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ==============================
# 2. CONVERT SENTIMENT LABELS
# ==============================

df["sentiment"] = df["sentiment"].replace({
    0: "Negative",
    4: "Positive"
})


# ==============================
# 3. CLEAN TWEETS
# ==============================

def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = re.sub(r"#", "", text)

    # Remove special characters
    text = re.sub(r"[^A-Za-z\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning tweets...")

df["clean_text"] = df["text"].apply(clean_text)

print("Cleaning completed!")


# ==============================
# 4. USE 200,000 TWEETS
# ==============================

df = df.sample(
    n=200000,
    random_state=42
)

print("\nUsing dataset size:", df.shape)


# ==============================
# 5. SPLIT DATA
# ==============================

X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 6. TF-IDF
# ==============================

print("\nConverting text into numerical features using TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")


# ==============================
# 7. TRAIN MODEL
# ==============================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==============================
# 8. PREDICTION
# ==============================

y_pred = model.predict(X_test_tfidf)


# ==============================
# 9. MODEL EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==============================
# 10. CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Negative", "Positive"]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Sentiment Analysis Confusion Matrix")

plt.tight_layout()

plt.savefig("graphs/confusion_matrix.png")

plt.show()


# ==============================
# 11. SAVE MODEL
# ==============================

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved as:")
print("sentiment_model.pkl")

print("TF-IDF vectorizer saved as:")
print("tfidf_vectorizer.pkl")

print("\nPROJECT TRAINING COMPLETED!")