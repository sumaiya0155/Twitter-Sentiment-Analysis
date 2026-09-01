import streamlit as st
import joblib
import re


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)


# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ==============================
# TEXT CLEANING
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


# ==============================
# HEADER
# ==============================

st.title("💬 Twitter Sentiment Analyzer")

st.write(
    "Analyze the sentiment of a tweet using "
    "Natural Language Processing and Machine Learning."
)

st.markdown("---")


# ==============================
# INPUT
# ==============================

st.subheader("📝 Enter Your Tweet")

user_text = st.text_area(
    "Type your tweet or any text:",
    height=150,
    placeholder="Example: I absolutely love this product!"
)


# ==============================
# ANALYZE BUTTON
# ==============================

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not user_text.strip():

        st.warning("⚠️ Please enter some text first.")

    else:

        # Clean text
        cleaned_text = clean_text(user_text)

        # Convert text to TF-IDF
        text_vector = vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(text_vector)[0]

        # Probabilities
        probabilities = model.predict_proba(text_vector)[0]

        negative_probability = probabilities[
            list(model.classes_).index("Negative")
        ]

        positive_probability = probabilities[
            list(model.classes_).index("Positive")
        ]


        # ==============================
        # RESULT
        # ==============================

        st.markdown("---")
        st.subheader("📊 Prediction Result")


        if prediction == "Positive":

            st.success("😊 POSITIVE SENTIMENT")

        else:

            st.error("😞 NEGATIVE SENTIMENT")


        # ==============================
        # CONFIDENCE
        # ==============================

        confidence = max(
            positive_probability,
            negative_probability
        ) * 100

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )


        # ==============================
        # PROBABILITY
        # ==============================

        st.subheader("📈 Sentiment Probability")

        col1, col2 = st.columns(2)

        with col1:

            st.write("😞 Negative")

            st.progress(
                int(negative_probability * 100)
            )

            st.write(
                f"{negative_probability * 100:.2f}%"
            )


        with col2:

            st.write("😊 Positive")

            st.progress(
                int(positive_probability * 100)
            )

            st.write(
                f"{positive_probability * 100:.2f}%"
            )


# ==============================
# EXAMPLES
# ==============================

st.markdown("---")

st.subheader("💡 Example Tweets")

st.write("😊 Positive:")
st.code("I absolutely love this product! It is amazing.")

st.write("😞 Negative:")
st.code("This is the worst service ever. I hate it.")


# ==============================
# PROJECT INFORMATION
# ==============================

st.markdown("---")

st.subheader("ℹ️ About the Project")

st.write(
    """
    This project uses Natural Language Processing (NLP)
    and Machine Learning to classify text into Positive
    or Negative sentiment.

    Model: Logistic Regression

    Feature Extraction: TF-IDF

    Dataset: Sentiment140
    """
)


# ==============================
# FOOTER
# ==============================

st.markdown("---")

st.caption(
    "Twitter Sentiment Analysis | NLP + Machine Learning"
)