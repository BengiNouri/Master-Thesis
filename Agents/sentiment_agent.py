import os
import time
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import pipeline
from datetime import datetime
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 🔐 Initialize Firebase
# ─────────────────────────────────────────────────────────────────────────────
def initialize_firebase() -> firestore.Client:
    paths = [
        r"C:\MasterThesis\Keys.json",
        r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
        r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
    ]
    if not firebase_admin._apps:
        for p in paths:
            if os.path.exists(p):
                cred = credentials.Certificate(p)
                firebase_admin.initialize_app(cred)
                break
        else:
            raise FileNotFoundError("No Firebase credentials found.")
    return firestore.client()

db = initialize_firebase()

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️ Initialize FinBERT Sentiment Analyzer
# ─────────────────────────────────────────────────────────────────────────────
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ─────────────────────────────────────────────────────────────────────────────
# 🔄 Retry decorator for analysis crashes
# ─────────────────────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception)
)
def analyze_snippet(snippet: str):
    return sentiment_analyzer(snippet)[0]

# ─────────────────────────────────────────────────────────────────────────────
# 🔍 Analyze & Store Sentiment in Batches
# ─────────────────────────────────────────────────────────────────────────────
def analyze_sentiment_and_store(batch_size: int = 500):
    news_ref = db.collection("news")
    docs = news_ref.stream()
    batch = db.batch()
    count = 0

    for doc in docs:
        data = doc.to_dict()
        doc_id = doc.id

        # Skip if already analyzed
        if data.get("sentiment_label") is not None and data.get("sentiment_score") is not None:
            continue

        content = (data.get("content") or "").strip()
        if not content:
            print(f"⚠️ Skipping empty content for {doc_id}")
            continue

        snippet = content[:512]
        try:
            result = analyze_snippet(snippet)
            label = result.get("label", "Neutral").capitalize()
            score = round(result.get("score", 0.0), 4)
        except Exception as e:
            print(f"❌ Error analyzing sentiment for {doc_id}: {e}")
            continue

        batch.update(news_ref.document(doc_id), {
            "sentiment_label": label,
            "sentiment_score": score,
            "analyzed_at":     datetime.utcnow().isoformat() + "Z"
        })
        count += 1

        if count % batch_size == 0:
            batch.commit()
            batch = db.batch()

    if count % batch_size != 0:
        batch.commit()

    print(f"✅ Sentiment updated for {count} articles.")

# ─────────────────────────────────────────────────────────────────────────────
# ✔️ Verify Unprocessed Articles
# ─────────────────────────────────────────────────────────────────────────────
def verify_sentiment_mapping():
    missing = []
    for doc in db.collection("news").stream():
        d = doc.to_dict()
        if d.get("sentiment_label") is None or d.get("sentiment_score") is None:
            missing.append(doc.id)
    if missing:
        print(f"⚠️ Missing sentiment for IDs: {missing}")
    else:
        print("✅ All news articles have sentiment data.")

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Migration Utility (optional)
# ─────────────────────────────────────────────────────────────────────────────
def migrate_sentiment():
    sent_ref = db.collection("sentiment_analysis").stream()
    merged = 0
    for sdoc in sent_ref:
        sdata = sdoc.to_dict()
        nid = sdata.get("news_id")
        if not nid:
            continue
        try:
            db.collection("news").document(nid).update({
                "sentiment_label": sdata.get("label","Neutral"),
                "sentiment_score": sdata.get("score",0.0),
                "analyzed_at":     sdata.get("analyzed_at")
            })
            merged += 1
        except Exception as e:
            print(f"❌ Migration error for {nid}: {e}")
    print(f"🔗 Migrated {merged} sentiment records.")

# ─────────────────────────────────────────────────────────────────────────────
# 🏁 Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyze_sentiment_and_store()
    verify_sentiment_mapping()
    # migrate_sentiment()  # uncomment to run migrations
