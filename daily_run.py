import os
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials

# External data sources
import yfinance as yf
from tiingo import TiingoClient

# Agents
from Agents.news_agent import process_articles
from Agents.rag_agent import generate_rag_response
from Agents.sentiment_agent import analyze_sentiment_and_store, migrate_sentiment

import warnings
import tensorflow as tf
import torch
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ─────────────────────────────────────────────────────────────────────────────
# Configurations
# ─────────────────────────────────────────────────────────────────────────────
VERBOSE = True
EXPERIMENT_START_DATE = datetime(2025, 1, 22)
RETRY_SLEEP_SECONDS = 60
TIINGO_BACKFILL_DAYS = 7

# ─────────────────────────────────────────────────────────────────────────────
# Load environment & init clients
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("❌ OPENAI_API_KEY not set.")

# Tiingo
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise RuntimeError("❌ TIINGO_API_KEY not set in environment.")
_tiingo_client = TiingoClient({"api_key": TIINGO_API_KEY, "session": True})

# ─────────────────────────────────────────────────────────────────────────────
# Suppress TF/torch warnings
# ─────────────────────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Firebase
# ─────────────────────────────────────────────────────────────────────────────
def initialize_firebase():
    paths = [
        r"C:\MasterThesis\Keys.json",
        r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
        r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
    ]
    if not firebase_admin._apps:
        for p in paths:
            if os.path.exists(p):
                cred = credentials.Certificate(p)
                break
        else:
            raise FileNotFoundError("Firebase credentials not found.")
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = initialize_firebase()

# ─────────────────────────────────────────────────────────────────────────────
# Price fetching: Tiingo first, then yfinance
# ─────────────────────────────────────────────────────────────────────────────
class PriceFetchError(Exception): pass

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def _fetch_with_yf(ticker):
    data = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
    if len(data) < 2:
        raise PriceFetchError(f"Insufficient yfinance data for {ticker}")
    return float(data.iloc[-1]), float(data.iloc[-2])


def _fetch_with_tiingo(ticker):
    end = datetime.utcnow().date()
    start = end - timedelta(days=TIINGO_BACKFILL_DAYS)
    prices = _tiingo_client.get_ticker_price(
        ticker,
        startDate=start.isoformat(),
        endDate=end.isoformat(),
        frequency='daily'
    )
    if not prices or len(prices) < 2:
        raise PriceFetchError(f"Insufficient Tiingo data for {ticker}")
    latest = prices[-1]['adjClose']
    prev = prices[-2]['adjClose']
    return float(latest), float(prev)


def fetch_closing_prices(ticker):
    """
    Primary: Tiingo; fallback: yfinance
    """
    try:
        return _fetch_with_tiingo(ticker)
    except Exception as ti:
        if VERBOSE:
            print(f"⚠️ Tiingo failed for {ticker}: {ti}. Falling back to yfinance...")
    try:
        return _fetch_with_yf(ticker)
    except Exception as yf:
        raise PriceFetchError(f"Both price sources failed for {ticker}: ti={ti}, yf={yf}")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: link
# ─────────────────────────────────────────────────────────────────────────────
def link_news_to_economic_data(news_id, stock):
    db.collection("latest_economic_data").document(stock).update({
        "linked_news_ids": firestore.ArrayUnion([news_id])
    })
    db.collection("news").document(news_id).update({"economic_data_id": stock})

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate recommendation correctness
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(stock, rec, model_name, latest, prev):
    moved = latest > prev
    correct = (rec.lower() in ["buy","hold"] and moved) or (rec.lower()=="sell" and not moved)
    arrow = "↑" if moved else "↓"
    pct = (latest - prev) / prev * 100 if prev else 0
    print(f"{stock} [{model_name}]: {prev:.2f}→{latest:.2f} ({arrow}{pct:.2f}%) rec={rec} → {correct}")
    return correct

# ─────────────────────────────────────────────────────────────────────────────
# Store recommendation + economic data
# ─────────────────────────────────────────────────────────────────────────────
def store_recommendation(stock, agg_rec, gpt_rec, summary, corr, latest, prev, detail):
    experiment_day = min((datetime.now() - EXPERIMENT_START_DATE).days, 90)
    # Build doc matching model_recommendations schema
    rec_doc = {
        "stock_ticker": stock,
        "sentiment_summary": summary,
        "latest_close": latest,
        "previous_close": prev,
        "is_correct": corr,
        "aggregator_recommendation": agg_rec,
        "gpt_recommendation": gpt_rec,
        "timestamp": datetime.now().isoformat(),
        "experiment_day": experiment_day,
        "recommendation_detail": detail
    }
    rec_ref = db.collection("model_recommendations").add(rec_doc)
    rec_id = rec_ref[1].id
    print(f"✅ Stored rec {rec_id} for {stock}")

    econ_doc = {
        "stock_ticker": stock,
        "experiment_day": experiment_day,
        "latest_close": latest,
        "previous_close": prev,
        "price_change": (latest - prev) / prev * 100 if prev else None,
        "sentiment_summary": summary,
        "recommendation_id": rec_id,
        "timestamp": datetime.now().isoformat()
    }
    db.collection("economic_data").add(econ_doc)
    print(f"✅ Stored econ for {stock}")

# ─────────────────────────────────────────────────────────────────────────────
# Daily pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_daily_pipeline(stocks, articles_per_stock=20):
    today = datetime.utcnow().date()
    for stock in stocks:
        print(f"\n--- Processing {stock} ---")
        process_articles([stock], articles_per_stock)
        # link
        for d in db.collection("news").where("keywords","array_contains",stock).stream():
            if not d.to_dict().get("economic_data_id"):
                link_news_to_economic_data(d.id, stock)
        # sentiment
        analyze_sentiment_and_store()
        migrate_sentiment()
        # gather today's articles
        docs = []
        for snap in db.collection("news").where("economic_data_id","==",stock).stream():
            data = snap.to_dict()
            ts = data.get("timestamp")
            if not ts or "sentiment_score" not in data:
                continue
            try:
                dt = datetime.fromisoformat(ts).date()
            except:
                continue
            if dt == today:
                docs.append(data)
        if not docs:
            print(f"⚠️ No processed articles for {stock} today. Skipping.")
            continue
        # print sentiment
        print(f"Today's sentiment scores for {stock}:")
        for d in docs:
            print(f"  • {d.get('title','[no title]')} → {d['sentiment_score']:.3f} ({d['sentiment_label']})")
        # generate recs
        agg_rec, gpt_rec, detail, summary = generate_rag_response(f"Outlook for {stock}?", docs)
        # fetch prices
        try:
            latest, prev = fetch_closing_prices(stock)
        except PriceFetchError as e:
            print(f"❌ Could not fetch prices for {stock}: {e}")
            continue
        # evaluate
        corr = evaluate_model(stock, gpt_rec, "GPT", latest, prev)
        # store
        store_recommendation(stock, agg_rec, gpt_rec, summary, corr, latest, prev, detail)
    print("\n✅ Daily pipeline complete.")

if __name__ == "__main__":
    run_daily_pipeline(["TSLA","AAPL","MSFT","NVDA","NVO"])
