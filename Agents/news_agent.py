import os
import sys
import time
import requests
from datetime import datetime
import firebase_admin
from firebase_admin import firestore, credentials
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Suppress TensorFlow and Torch warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Add project root to sys.path if needed for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────────────────────────────────────
# 🔐 Initialize Firebase
# ─────────────────────────────────────────────────────────────────────────────

def initialize_firebase():
    primary = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    if not firebase_admin._apps:
        cred_path = primary if os.path.exists(primary) else fallback
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"Firebase credentials not found at {cred_path}")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = initialize_firebase()

# ─────────────────────────────────────────────────────────────────────────────
# ✅ Build Stock Mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_stock_mapping():
    mapping = {}
    try:
        for doc in db.collection("latest_economic_data").stream():
            data = doc.to_dict()
            ticker = doc.id.upper().strip()
            name = data.get("long_name", "").lower().strip()
            if ticker:
                mapping[ticker] = ticker
            if name:
                mapping[name] = ticker
        print("✅ Stock mapping loaded.")
    except Exception as e:
        print(f"❌ Error loading stock mapping: {e}")
    return mapping

STOCK_MAPPING = build_stock_mapping()

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 NewsAPI fetch with retry/backoff
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_news_articles(keywords, page_size=10):
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY not set")
    query = " OR ".join(keywords)
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size
    }
    resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("articles", [])

# ─────────────────────────────────────────────────────────────────────────────
# 🔍 Deduplicate by URL
# ─────────────────────────────────────────────────────────────────────────────

def url_exists(url: str) -> bool:
    snaps = db.collection("news").where("url", "==", url).limit(1).get()
    return len(snaps) > 0

# ─────────────────────────────────────────────────────────────────────────────
# 📄 Process & store articles in batches
# ─────────────────────────────────────────────────────────────────────────────

def process_articles(keywords, page_size=10):
    print(f"🔎 Fetching news for: {keywords}")
    try:
        raw = fetch_news_articles(keywords, page_size)
    except Exception as e:
        print(f"❌ NewsAPI error: {e}")
        return

    if not raw:
        print("⚠️ No articles returned from NewsAPI.")
        return

    batch = db.batch()
    count = 0

    for art in raw:
        url = art.get("url")
        if not url or url_exists(url):
            continue

        tag = keywords[0].strip().upper()
        econ_id = STOCK_MAPPING.get(tag.lower(), tag)

        doc_ref = db.collection("news").document()
        payload = {
            "title": art.get("title"),
            "content": art.get("content"),
            "url": url,
            "publishedAt": art.get("publishedAt"),
            "source": art.get("source", {}).get("name"),
            "keywords": [k.lower().strip() for k in keywords],
            "economic_data_id": econ_id,
            "sentiment_label": None,
            "sentiment_score": None,
            "analyzed_at": None
        }
        batch.set(doc_ref, payload)
        count += 1

        if count % 500 == 0:
            batch.commit()
            batch = db.batch()

    if count % 500 != 0:
        batch.commit()

    print(f"✅ Stored {count} new articles in 'news'.")

if __name__ == "__main__":
    process_articles(["Tesla", "TSLA"], page_size=5)
