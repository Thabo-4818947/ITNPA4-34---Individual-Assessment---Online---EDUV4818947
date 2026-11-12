
import re
import joblib
import nltk
import streamlit as st

from pathlib import Path
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english")) - {"no", "not", "nor"}

model_path = Path(__file__).parent / "best_research_classifier.pkl"
model = joblib.load(model_path)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)

st.set_page_config(
    page_title="Research Abstract Classifier",
    page_icon="📄"
)

st.title("Researchly Research Abstract Classifier")

st.write(
    "Classify research abstracts as Artificial Intelligence "
    "or Classical Research."
)

abstract = st.text_area(
    "Enter your research abstract:",
    height=250
)

if st.button("Classify Abstract"):
    if not abstract.strip():
        st.warning("Please enter a research abstract.")
    else:
        cleaned_abstract = clean_text(abstract)
        prediction = int(model.predict([cleaned_abstract])[0])

        if prediction == 1:
            st.success("Prediction: ARTIFICIAL INTELLIGENCE (AI)")
        else:
            st.success("Prediction: CLASSICAL RESEARCH")
