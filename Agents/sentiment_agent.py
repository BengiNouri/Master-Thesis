import os
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import pipeline
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables (make sure you have a .env file if needed)
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 🔐 Initialize Firebase
# ─────────────────────────────────────────────────────────────────────────────

def initialize_firebase():
    """
    Initialize Firebase with fallback credential paths.
    """
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

db = initialize_firebase()

# ─────────────────────────────────────────────────────────────────────────────
# ✅ Initialize FinBERT Sentiment Analyzer
# ─────────────────────────────────────────────────────────────────────────────

sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ─────────────────────────────────────────────────────────────────────────────
# ✅ Analyze Sentiment & Update Firestore "news" Collection
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sentiment_and_store():
    """
    1. Fetch 'news' docs that do NOT have sentiment fields.
    2. Run FinBERT on each doc's 'content'.
    3. Update the doc with 'sentiment_label', 'sentiment_score', and 'analyzed_at'.
    """
    try:
        news_ref = db.collection("news")
        docs = news_ref.stream()
        
        for doc in docs:
            news_data = doc.to_dict()
            news_id = doc.id

            # Skip if sentiment data exists
            if "sentiment_label" in news_data and "sentiment_score" in news_data:
                continue

            content = news_data.get("content", "").strip()
            if not content:
                print(f"⚠️ Skipping empty content for news ID: {news_id}")
                continue

            try:
                # Use up to 512 characters (or tokens) for analysis
                snippet = content if len(content) < 512 else content[:512]
                result = sentiment_analyzer(snippet)[0]
                label = result.get("label", "Neutral").capitalize()
                score = round(result.get("score", 0.0), 4)

                # Update the news document
                news_ref.document(news_id).update({
                    "sentiment_label": label,
                    "sentiment_score": score,
                    "analyzed_at": datetime.now().isoformat()
                })

                print(f"✅ Updated sentiment for news ID {news_id}: {label} ({score})")

            except Exception as sentiment_error:
                print(f"❌ Error analyzing sentiment for news ID {news_id}: {sentiment_error}")

    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ✅ Verify News Sentiment (Optional)
# ─────────────────────────────────────────────────────────────────────────────

def verify_sentiment_mapping():
    """
    Check news docs that lack sentiment data and print their IDs.
    """
    try:
        missing_sentiments = []
        news_docs = db.collection("news").stream()
        for news_doc in news_docs:
            data = news_doc.to_dict()
            if "sentiment_label" not in data or "sentiment_score" not in data:
                missing_sentiments.append(news_doc.id)
        if missing_sentiments:
            print(f"⚠️ Missing sentiment data for news IDs: {missing_sentiments}")
        else:
            print("✅ All news articles have sentiment data.")
    except Exception as e:
        print(f"❌ Error verifying sentiment mapping: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ✅ Migrate Sentiment from 'sentiment_analysis' to 'news'
# ─────────────────────────────────────────────────────────────────────────────

def migrate_sentiment():
    """
    For each doc in 'sentiment_analysis':
      - Find the corresponding 'news' document by 'news_id'
      - Update that 'news' document with the sentiment fields.
    """
    sentiment_ref = db.collection("sentiment_analysis")
    docs = sentiment_ref.stream()
    merged_count = 0
    for sdoc in docs:
        sdata = sdoc.to_dict()
        news_id = sdata.get("news_id")
        if not news_id:
            print(f"❌ No 'news_id' for sentiment doc {sdoc.id}, skipping.")
            continue

        try:
            label = sdata.get("label", "Neutral")
            score = sdata.get("score", 0.0)
            analyzed_at = sdata.get("analyzed_at", None)

            db.collection("news").document(news_id).update({
                "sentiment_label": label,
                "sentiment_score": score,
                "analyzed_at": analyzed_at
            })
            merged_count += 1
            print(f"✅ Merged sentiment into news doc {news_id} ({label}, {score})")
        except Exception as e:
            print(f"❌ Error merging doc {sdoc.id}: {e}")

    if merged_count == 0:
        print("⚠️ No leftover sentiment docs found, or all had missing news_id.")
    else:
        print(f"🔗 Merged {merged_count} docs from 'sentiment_analysis' into 'news'!")

if __name__ == "__main__":
    analyze_sentiment_and_store()
    verify_sentiment_mapping()
