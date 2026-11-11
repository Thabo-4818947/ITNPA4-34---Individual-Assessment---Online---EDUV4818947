import re
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english")) - {"no", "not", "nor"}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

st.set_page_config(page_title="Abstract Classifier", page_icon="📄")

model_path = Path(__file__).parent / "balanced_svm_model.pkl"
model = joblib.load(model_path)

st.title("Research Abstract Classifier")
st.write("Classify a research abstract as AI-generated or Classical.")

abstract = st.text_area(
    "Paste the research abstract below:",
    height=250,
    placeholder="Enter the abstract text here...",
)

if st.button("Classify"):
    if not abstract.strip():
        st.warning("Please enter an abstract first.")
    else:
        cleaned_abstract = clean_text(abstract)
        prediction = int(model.predict([cleaned_abstract])[0])

        if prediction == 1:
            st.subheader("Prediction: AI")
        else:
            st.subheader("Prediction: Classical")
