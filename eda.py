import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# LOAD DATASET
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


# ==============================
# CONVERT LABELS
# ==============================

df["sentiment"] = df["sentiment"].replace({
    0: "Negative",
    4: "Positive"
})


print("Dataset loaded successfully!")
print("\nSentiment counts:")
print(df["sentiment"].value_counts())


# ==============================
# SENTIMENT DISTRIBUTION
# ==============================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="sentiment"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")

plt.tight_layout()

plt.savefig("graphs/sentiment_distribution.png")

plt.show()


print("\nGraph saved successfully!")
print("graphs/sentiment_distribution.png")