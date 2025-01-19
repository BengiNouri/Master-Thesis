import os
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import pipeline
from datetime import datetime

# Initialize Firebase
def initialize_firebase():
    vm_path = r"C:\MasterThesis\Keys.json"
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    if not firebase_admin._apps:
        cred = None
        if os.path.exists(vm_path):
            cred = credentials.Certificate(vm_path)
        elif os.path.exists(primary_path):
            cred = credentials.Certificate(primary_path)
        elif os.path.exists(fallback_path):
            cred = credentials.Certificate(fallback_path)
        else:
            raise FileNotFoundError("Firebase credentials file not found in any path.")
        
        firebase_admin.initialize_app(cred)

    return firestore.client()

# Initialize Firebase client
db = initialize_firebase()

# Initialize FinBERT Sentiment Analyzer
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment_and_store():
    """
    Analyze sentiment of unprocessed news articles and link the result back to the news collection.
    """
    try:
        news_ref = db.collection("news")
        docs = news_ref.where("sentiment_id", "==", None).stream()  # Only analyze if sentiment is missing

        for doc in docs:
            news_data = doc.to_dict()
            news_id = doc.id
            content = news_data.get("content", "")

            if not content:
                print(f"⚠️ Skipping empty content for news ID: {news_id}")
                continue

            try:
                # Perform sentiment analysis
                result = sentiment_analyzer(content[:512])[0]
                sentiment_label = result["label"].capitalize()
                sentiment_score = round(result["score"], 4)

                # Store sentiment in 'sentiment_analysis' collection
                sentiment_doc = {
                    "news_id": news_id,
                    "label": sentiment_label,
                    "score": sentiment_score,
                    "analyzed_at": datetime.now().isoformat()
                }
                sentiment_ref = db.collection("sentiment_analysis").add(sentiment_doc)
                sentiment_id = sentiment_ref[1].id  # Get the generated sentiment document ID

                # Update the news document with sentiment_id
                news_ref.document(news_id).update({
                    "sentiment_id": sentiment_id
                })

                print(f"✅ Sentiment stored for news ID: {news_id} | {sentiment_label} ({sentiment_score})")

            except Exception as sentiment_error:
                print(f"❌ Error analyzing sentiment for news ID {news_id}: {sentiment_error}")

        print("✅ Sentiment analysis and linking completed.")

    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")

# 🚀 Run sentiment analysis if the script is executed directly
if __name__ == "__main__":
    analyze_sentiment_and_store()
