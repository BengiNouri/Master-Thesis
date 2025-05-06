import os
import openai
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials

# Agents
from Agents.news_agent import process_articles, STOCK_MAPPING
from Agents.rag_agent import fetch_closing_prices, generate_rag_response
from Agents.sentiment_agent import analyze_sentiment_and_store, migrate_sentiment

import warnings
import tensorflow as tf
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Configurations
# ─────────────────────────────────────────────────────────────────────────────
VERBOSE = True
EXPERIMENT_START_DATE = datetime(2025, 1, 22)

# ─────────────────────────────────────────────────────────────────────────────
# Load env & init OpenAI
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("❌ OPENAI_API_KEY not set.")

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
    vm_path       = r"C:\MasterThesis\Keys.json"
    primary_path  = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    if not firebase_admin._apps:
        if   os.path.exists(vm_path):       cred = credentials.Certificate(vm_path)
        elif os.path.exists(primary_path):  cred = credentials.Certificate(primary_path)
        elif os.path.exists(fallback_path): cred = credentials.Certificate(fallback_path)
        else: raise FileNotFoundError("Firebase credentials not found.")
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = initialize_firebase()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Link news → economic_data document
# ─────────────────────────────────────────────────────────────────────────────
def link_news_to_economic_data(news_id: str, stock_ticker: str):
    db.collection("latest_economic_data") \
      .document(stock_ticker) \
      .update({"linked_news_ids": firestore.ArrayUnion([news_id])})
    db.collection("news").document(news_id).update({"economic_data_id": stock_ticker})

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate Recommendation Correctness
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_recommendation(stock_ticker, recommendation):
    latest, prev = fetch_closing_prices(stock_ticker)
    if latest is None or prev is None:
        if VERBOSE:
            print(f"⚠️ Missing price data for {stock_ticker}.")
        return False, latest, prev

    moved_up = latest > prev
    is_correct = (
        (recommendation.lower() in ["buy","hold"] and moved_up) or
        (recommendation.lower()=="sell" and not moved_up)
    )
    if VERBOSE:
        arrow = "↑" if moved_up else "↓"
        pct = (latest - prev) / prev * 100
        print(f"{stock_ticker}: {prev:.2f}→{latest:.2f} ({arrow}{pct:.2f}%) rec={recommendation} → {is_correct}")
    return is_correct, latest, prev

# ─────────────────────────────────────────────────────────────────────────────
# Store Recommendation + Economic Data
# ─────────────────────────────────────────────────────────────────────────────
def store_recommendation(
    stock_ticker,
    aggregator_recommendation,
    gpt_recommendation,
    sentiment_summary,
    is_correct,
    latest_close,
    previous_close,
    full_text=""
):
    experiment_day = min((datetime.now() - EXPERIMENT_START_DATE).days, 90)
    price_change   = ((latest_close - previous_close) / previous_close * 100) if previous_close else 0

    # Model recommendations
    rec_ref = db.collection("model_recommendations").add({
        "stock_ticker":              stock_ticker,
        "aggregator_recommendation": aggregator_recommendation,
        "gpt_recommendation":        gpt_recommendation,
        "sentiment_summary":         sentiment_summary,
        "recommendation_detail":     full_text,
        "is_correct":                is_correct,
        "latest_close":              latest_close,
        "previous_close":            previous_close,
        "experiment_day":            experiment_day,
        "timestamp":                 datetime.now().isoformat()
    })
    rec_id = rec_ref[1].id

    # Economic data
    eco_ref = db.collection("economic_data").add({
        "stock_ticker":     stock_ticker,
        "experiment_day":   experiment_day,
        "latest_close":     latest_close,
        "previous_close":   previous_close,
        "price_change":     price_change,
        "sentiment_summary": sentiment_summary,
        "recommendation_id": rec_id,
        "timestamp":         datetime.now().isoformat()
    })
    eco_id = eco_ref[1].id

    print(f"✅ Stored rec {rec_id} & econ {eco_id} for {stock_ticker}")

# ─────────────────────────────────────────────────────────────────────────────
# Daily Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_daily_pipeline(stock_tickers, articles_per_stock=20):
    for stock in stock_tickers:
        print(f"\n--- Processing {stock} ---")

        # 1) Fetch news
        process_articles([stock], articles_per_stock)

        # 2) Link news → econ data
        snaps = db.collection("news").where("keywords", "array_contains", stock).stream()
        for doc in snaps:
            if not doc.to_dict().get("economic_data_id"):
                link_news_to_economic_data(doc.id, stock)

        # 3) Sentiment analysis + migrate leftovers
        analyze_sentiment_and_store()
        migrate_sentiment()

        # 4) Gather processed articles
        snaps2 = db.collection("news").where("economic_data_id", "==", stock).stream()
        news_docs = [d.to_dict() for d in snaps2 if "sentiment_label" in d.to_dict()]
        if not news_docs:
            print(f"⚠️ No processed articles for {stock}. Skipping.")
            continue

        # 5) Generate recommendations (now returns reasoning)
        agg, gpt, reasoning, summary = generate_rag_response(
            f"What’s the outlook for {stock}?", news_docs
        )

        # 6) Evaluate
        correct, latest, prev = evaluate_recommendation(stock, gpt)

        # 7) Store
        store_recommendation(stock, agg, gpt, summary, correct, latest, prev, full_text=reasoning)

    print("\n✅ Daily pipeline complete.")

if __name__ == "__main__":
    run_daily_pipeline(["TSLA", "AAPL", "MSFT", "NVDA", "NVO"])
