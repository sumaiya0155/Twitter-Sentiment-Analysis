import pandas as pd
import re
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score


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

print("Dataset loaded!")


# ==============================
# 2. CONVERT LABELS
# ==============================

df["sentiment"] = df["sentiment"].replace({
    0: "Negative",
    4: "Positive"
})


# ==============================
# 3. CLEAN TEXT
# ==============================

def clean_text(text):

    text = str(text)

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"[^A-Za-z\s]",
        "",
        text
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("Cleaning tweets...")

df["clean_text"] = df["text"].apply(clean_text)

print("Cleaning completed!")


# ==============================
# 4. SAMPLE DATA
# ==============================

df = df.sample(
    n=200000,
    random_state=42
)

print("Using 200,000 tweets.")


# ==============================
# 5. TRAIN / TEST SPLIT
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


# ==============================
# 6. TF-IDF
# ==============================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")


# ==============================
# 7. DEFINE MODELS
# ==============================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC()
}


# ==============================
# 8. TRAIN & COMPARE
# ==============================

results = {}

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")


for name, model in models.items():

    print(f"\nTraining {name}...")

    start_time = time.time()

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    training_time = time.time() - start_time

    results[name] = accuracy

    print(
        f"{name} Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )


# ==============================
# 9. DISPLAY RESULTS
# ==============================

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

for name, accuracy in results.items():

    print(
        f"{name}: "
        f"{accuracy * 100:.2f}%"
    )


# ==============================
# 10. BEST MODEL
# ==============================

best_model = max(
    results,
    key=results.get
)

print("\n==============================")

print(
    f"BEST MODEL: {best_model}"
)

print(
    f"BEST ACCURACY: "
    f"{results[best_model] * 100:.2f}%"
)

print("==============================")
import matplotlib.pyplot as plt

# Model names and accuracies
model_names = list(results.keys())
accuracies = list(results.values())

# Create graph
plt.figure(figsize=(8, 5))

plt.bar(model_names, accuracies)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

# Display accuracy values
for i, accuracy in enumerate(accuracies):
    plt.text(
        i,
        accuracy + 0.01,
        f"{accuracy * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()

# Save graph
plt.savefig("graphs/model_comparison.png")

plt.show()

print("\nModel comparison graph saved successfully!")