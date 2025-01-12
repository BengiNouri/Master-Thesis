import os
import sys
from transformers import pipeline
from firebase_admin import firestore, initialize_app, credentials
from dotenv import load_dotenv

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppresses INFO and WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Prevents oneDNN numerical differences
# Forbindelse til projektets rodmappe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Indlæs miljøvariabler
load_dotenv()

import firebase_admin
from firebase_admin import firestore, credentials

# Firebase Initialization
if not firebase_admin._apps:  # Correct: Check if Firebase is already initialized
    cred = credentials.Certificate(r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Opsummeringsmodel
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=50, min_length=20):
    """
    Opsummerer teksten ved hjælp af modellen.
    """
    if not text:
        return "No content to summarize."
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Error during summarization: {e}"

def summarize_documents_from_firestore():
    """
    Hent dokumenter fra Firestore, generér opsummeringer og gem dem tilbage.
    """
    try:
        # Hent nyhedsartikler fra Firestore
        news_ref = db.collection("news")
        articles = news_ref.stream()

        for article in articles:
            doc = article.to_dict()
            doc_id = article.id

            # Opsummering
            content = doc.get("content", "")
            if not content:
                print(f"No content for document ID: {doc_id}")
                continue

            summary = summarize_text(content)
            print(f"Document ID: {doc_id} | Summary: {summary}")

            # Gem opsummering tilbage i Firestore
            news_ref.document(doc_id).update({"summary": summary})
            print(f"Summary saved for document ID: {doc_id}")

    except Exception as e:
        print(f"Error summarizing documents: {e}")

if __name__ == "__main__":
    summarize_documents_from_firestore()
