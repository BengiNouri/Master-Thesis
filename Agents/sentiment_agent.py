import os
from firebase_admin import firestore, initialize_app, credentials
from dotenv import load_dotenv
from transformers import pipeline

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load environment variables
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
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        # Try the primary path first
        if os.path.exists(primary_path):
            cred = credentials.Certificate(primary_path)
        # If not, use the fallback path
        elif os.path.exists(fallback_path):
            cred = credentials.Certificate(fallback_path)
        else:
            raise FileNotFoundError("Firebase credentials file not found in both paths.")
        
        firebase_admin.initialize_app(cred)

    return firestore.client()

# Initialize Firebase
db = initialize_firebase()

# Initialize Sentiment Analysis Model with FinBERT
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def fetch_unprocessed_news():
    """
    Fetch news articles from Firestore that do not have sentiment analysis.
    """
    try:
        news_ref = db.collection("news")
        query = news_ref.where("sentiment_id", "==", None)  # Fetch articles without sentiment analysis
        return [doc for doc in query.stream()]
    except Exception as e:
        print(f"❌ Error fetching unprocessed news: {e}")
        return []

def analyze_sentiment_and_store():
    """
    Analyze sentiment of unprocessed news articles using FinBERT and store results in Firestore.
    """
    try:
        news_docs = fetch_unprocessed_news()

        if not news_docs:
            print("⚠️ No unprocessed news articles found.")
            return

        for news_doc in news_docs:
            news_data = news_doc.to_dict()
            news_id = news_doc.id
            content = news_data.get("content", "")

            if not content:
                print(f"⚠️ No content available for news article with ID: {news_id}")
                continue

            # Perform sentiment analysis with FinBERT
            sentiment_results = sentiment_analyzer(content[:512])  # Limit content length for processing
            sentiment_label = sentiment_results[0]["label"].lower()  # "positive", "neutral", "negative"
            sentiment_score = sentiment_results[0]["score"]

            # Store sentiment result in Firestore and link back to news with news_id
            sentiment_data = {
                "label": sentiment_label,
                "score": sentiment_score,
                "news_id": news_id  # Linking sentiment back to the news article
            }
            sentiment_ref = db.collection("sentiment").document()
            sentiment_ref.set(sentiment_data)
            sentiment_id = sentiment_ref.id

            # Update the news document with the sentiment_id
            db.collection("news").document(news_id).update({"sentiment_id": sentiment_id})

            print(f"✅ Processed sentiment for news ID: {news_id} | Sentiment: {sentiment_label} (Score: {sentiment_score})")

    except Exception as e:
        print(f"❌ Error in sentiment analysis: {e}")

if __name__ == "__main__":
    analyze_sentiment_and_store()
