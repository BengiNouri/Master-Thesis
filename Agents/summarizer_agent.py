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

import os
import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    """
    Initialize Firebase with a fallback if the primary path fails.
    """
    # Path for the virtual machine
    vm_path = r"C:\MasterThesis\Keys.json"
    
    # Local machine paths
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        # Try the VM path first
        if os.path.exists(vm_path):
            cred = credentials.Certificate(vm_path)
        # Try the primary local path
        elif os.path.exists(primary_path):
            cred = credentials.Certificate(primary_path)
        # Fallback local path
        elif os.path.exists(fallback_path):
            cred = credentials.Certificate(fallback_path)
        else:
            raise FileNotFoundError("Firebase credentials file not found in any path.")
        
        firebase_admin.initialize_app(cred)

    return firestore.client()


# Initialize Firebase
db = initialize_firebase()

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
    Generate and store summaries for news articles in the Firestore 'news' collection.
    """
    try:
        news_ref = db.collection("news")
        articles = news_ref.stream()

        for article in articles:
            doc = article.to_dict()
            doc_id = article.id
            content = doc.get("content", "")

            if not content:
                print(f"⚠️ No content for news ID: {doc_id}")
                continue

            # Generate summary
            summary = summarize_text(content)

            # Update the news document with the summary
            news_ref.document(doc_id).update({"summary": summary})
            print(f"✅ Summary stored for news ID: {doc_id}")

    except Exception as e:
        print(f"❌ Error summarizing documents: {e}")


if __name__ == "__main__":
    summarize_documents_from_firestore()
