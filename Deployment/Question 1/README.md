# Research Abstract Classifier

A Streamlit app that classifies research abstracts as AI-generated or Classical using a balanced SVM model.

## Files

- `classifier.py` — Streamlit web app
- `balanced_svm_model.pkl` — fitted TF-IDF and balanced SVM pipeline
- `requirements.txt` — package versions used by the app

## Run locally

Open Anaconda Prompt from the folder containing `deployment`, then run:

```bash
cd deployment
pip install -r requirements.txt
streamlit run classifier.py
```
