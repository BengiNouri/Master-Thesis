import os, sys, requests, time
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import firestore, credentials
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tensorflow as tf, torch

# ──────────────────────────────── house-keeping ─────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# let other modules import this package easily
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────── Firebase init ──────────────────────────────
def initialize_firebase() -> firestore.Client:
    cred_paths = [
        r"C:\MasterThesis\Keys.json",
        r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
        r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
    ]
    if not firebase_admin._apps:
        for p in cred_paths:
            if os.path.exists(p):
                firebase_admin.initialize_app(credentials.Certificate(p))
                break
        else:
            raise FileNotFoundError("❌ Firebase credentials not found.")
    return firestore.client()

db = initialize_firebase()

# ───────────────────────────── stock-ticker map ─────────────────────────────
def build_stock_mapping():
    mapping = {}
    for doc in db.collection("latest_economic_data").stream():
        d = doc.to_dict()
        mapping[doc.id.upper()]          = doc.id.upper()
        if name := d.get("long_name", "").lower().strip():
            mapping[name] = doc.id.upper()
    print("✅ Stock mapping loaded.")
    return mapping

STOCK_MAPPING = build_stock_mapping()

# ─────────────────────────── NewsAPI w/ retry logic ─────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_news_articles(keywords, page_size=10):
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY not set in environment.")
    query = " OR ".join(keywords)
    resp  = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size
        },
        timeout=10
    )
    resp.raise_for_status()
    return resp.json().get("articles", [])

# ─────────────────────────── main up-sert function ─────────────────────────
def process_articles(keywords, page_size: int = 10):
    print(f"🔎 Fetching news for: {keywords}")
    articles = fetch_news_articles(keywords, page_size)
    if not articles:
        print("⚠️  No articles returned from NewsAPI.")
        return

    new_count  = 0          # newly inserted docs
    up_count   = 0          # existing docs refreshed
    batch      = db.batch()

    for art in articles:
        url = art.get("url")
        if not url:
            continue

        now_iso = datetime.now(timezone.utc).isoformat()
        snap    = db.collection("news").where("url", "==", url).limit(1).get()

        # ── existing headline → refresh ingested_at only ───────────────────
        if snap:
            db.collection("news").document(snap[0].id).update({"ingested_at": now_iso})
            up_count += 1
            continue

        # ── brand-new headline → build full payload ────────────────────────
        tag     = keywords[0].strip().upper()
        econ_id = STOCK_MAPPING.get(tag.lower(), tag)

        payload = {
            "title":          art.get("title"),
            "content":        art.get("content"),
            "url":            url,
            "timestamp":      art.get("publishedAt"),       # original pub-date
            "ingested_at":    now_iso,                      # first seen now
            "source":         art.get("source", {}).get("name"),
            "keywords":       [k.lower().strip() for k in keywords],
            "economic_data_id": econ_id,
            "sentiment_label":  None,
            "sentiment_score":  None,
            "analyzed_at":      None
        }
        doc_ref = db.collection("news").document()
        batch.set(doc_ref, payload)
        new_count += 1

        if new_count % 500 == 0:
            batch.commit()
            batch = db.batch()

    if new_count % 500 != 0:
        batch.commit()

    print(f"✅ Stored {new_count} new article(s); "
          f"🕑 refreshed {up_count} existing doc(s).")

# ──────────────────────────────── CLI helper ────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("keywords", nargs="*", help="keywords to search (quote multi-word)")
    ap.add_argument("--page", type=int, default=5, help="NewsAPI page_size")
    args = ap.parse_args()

    # default to Tesla if user supplied nothing
    kw = args.keywords or ["Tesla", "TSLA"]
    process_articles(kw, page_size=args.page)
