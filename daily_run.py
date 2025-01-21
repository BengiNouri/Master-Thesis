import os
import openai
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials

# Agents
from Agents.news_agent import process_articles, link_news_to_economic_data
from Agents.rag_agent import generate_rag_response
from Agents.sentiment_agent import analyze_sentiment_and_store

import warnings
import tensorflow as tf
import torch

# ───────────────────────────────────────────────────────────────────────────────
# ✅ Load environment variables
# ───────────────────────────────────────────────────────────────────────────────

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ───────────────────────────────────────────────────────────────────────────────
# ✅ Suppress Warnings
# ───────────────────────────────────────────────────────────────────────────────

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

# ───────────────────────────────────────────────────────────────────────────────
# ✅ Define experiment start date (for correctness tracking)
# ───────────────────────────────────────────────────────────────────────────────

EXPERIMENT_START_DATE = datetime(2025, 1, 18)

# ───────────────────────────────────────────────────────────────────────────────
# 🔐 Initialize Firebase
# ───────────────────────────────────────────────────────────────────────────────

def initialize_firebase():
    """
    Initialize Firebase and return a Firestore client.
    """
    vm_path = r"C:\MasterThesis\Keys.json"
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    if not firebase_admin._apps:
        if os.path.exists(vm_path):
            cred = credentials.Certificate(vm_path)
        elif os.path.exists(primary_path):
            cred = credentials.Certificate(primary_path)
        elif os.path.exists(fallback_path):
            cred = credentials.Certificate(fallback_path)
        else:
            raise FileNotFoundError("Firebase credentials file not found.")
        firebase_admin.initialize_app(cred)

    return firestore.client()

db = initialize_firebase()

# ───────────────────────────────────────────────────────────────────────────────
# ✅ Migrate Sentiment from 'sentiment_analysis' → 'news'
# ───────────────────────────────────────────────────────────────────────────────

def migrate_sentiment():
    """
    For each doc in 'sentiment_analysis':
      - find the corresponding 'news' doc by 'news_id'
      - copy 'label', 'score', 'analyzed_at' into that 'news' document
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

            # Update the matching 'news' doc
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

# ───────────────────────────────────────────────────────────────────────────────
# 📈 Fetch Stock Prices
# ───────────────────────────────────────────────────────────────────────────────

def fetch_closing_prices(stock_ticker):
    """
    Fetch the latest and previous closing prices for a stock (5-day range).
    """
    try:
        ticker = yf.Ticker(stock_ticker)
        hist = ticker.history(period="5d")
        latest_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        return latest_close, previous_close
    except Exception as e:
        print(f"❌ Error fetching stock data for {stock_ticker}: {e}")
        return None, None

# ───────────────────────────────────────────────────────────────────────────────
# 📊 Evaluate Recommendation
# ───────────────────────────────────────────────────────────────────────────────

def evaluate_recommendation(stock_ticker, recommendation):
    """
    Compare today's close vs. yesterday's close to see if the recommendation 
    was 'correct' for the movement. (Buy/Hold implies up; Sell implies down).
    """
    latest_close, previous_close = fetch_closing_prices(stock_ticker)
    if not latest_close or not previous_close:
        print(f"⚠️ Missing stock price data for {stock_ticker}. "
              f"Latest: {latest_close}, Previous: {previous_close}")
        return False, latest_close, previous_close

    price_movement = "up" if latest_close > previous_close else "down"
    is_correct = (
        (recommendation.lower() in ["buy", "hold"] and price_movement == "up")
        or
        (recommendation.lower() == "sell" and price_movement == "down")
    )
    
    print(f"🔄 Stock Price for {stock_ticker}: "
          f"Latest={latest_close}, Previous={previous_close}, Movement={price_movement}")
    print(f"🔍 Recommendation Correctness: {is_correct}")
    
    return is_correct, latest_close, previous_close

# ───────────────────────────────────────────────────────────────────────────────
# 📝 Store Recommendation Results
# ───────────────────────────────────────────────────────────────────────────────

def store_recommendation(stock_ticker, short_rec, full_detail, is_correct, latest_close, previous_close):
    """
    Store two fields:
      1. recommendation_summary: "Buy", "Sell", or "Hold"
      2. recommendation_detail: your multi-paragraph explanation
    """
    try:
        experiment_day = (datetime.now() - EXPERIMENT_START_DATE).days
        experiment_day = min(experiment_day, 90)

        db.collection("model_recommendations").add({
            "stock_ticker": stock_ticker,
            "recommendation_summary": short_rec,   # single word
            "recommendation_detail": full_detail,  # big text
            "is_correct": is_correct,
            "latest_close": latest_close,
            "previous_close": previous_close,
            "timestamp": datetime.now().isoformat(),
            "experiment_day": experiment_day
        })
        print(f"✅ Stored recommendation for {stock_ticker}: {short_rec} | Correct: {is_correct}")
    except Exception as e:
        print(f"❌ Error storing recommendation: {e}")


# ───────────────────────────────────────────────────────────────────────────────
# ✅ Check if Article Was Processed
# ───────────────────────────────────────────────────────────────────────────────

def is_article_processed(article_id):
    """
    Return True if a 'news' doc has 'sentiment_label' & 'sentiment_score'.
    """
    try:
        doc_snapshot = db.collection("news").document(article_id).get()
        if not doc_snapshot.exists:
            return False
        data = doc_snapshot.to_dict()
        return ("sentiment_label" in data and "sentiment_score" in data)
    except Exception as e:
        print(f"❌ Error checking if article {article_id} is processed: {e}")
        return False

# ───────────────────────────────────────────────────────────────────────────────
# 🔄 Run Daily Pipeline
# ───────────────────────────────────────────────────────────────────────────────

def run_daily_pipeline(stock_tickers, articles_per_stock=5):
    """
    Steps:
      1) Fetch & store new articles (process_articles).
      2) Link them to the correct ticker (link_news_to_economic_data).
      3) Run in-place sentiment analysis (analyze_sentiment_and_store).
      4) [MIGRATE] Merge leftover docs from 'sentiment_analysis' => 'news'.
      5) Fetch newly processed articles (with sentiment in 'news').
      6) Generate RAG recommendation (now returns short_rec, detail_text).
      7) Evaluate correctness & store recommendation (two fields).
    """
    for stock in stock_tickers:
        print(f"\n🔍 Processing {stock}...")

        # 1) Fetch & store new articles
        try:
            process_articles([stock], articles_per_stock)
            print(f"🔎 Fetched up to {articles_per_stock} articles for {stock}")
        except Exception as e:
            print(f"❌ Error processing articles for {stock}: {e}")
            continue

        # 2) Link docs to correct ticker
        try:
            newly_stored = db.collection("news") \
                .where("keywords", "array_contains", stock) \
                .stream()
            for doc in newly_stored:
                doc_data = doc.to_dict()
                if not doc_data.get("economic_data_id"):
                    link_news_to_economic_data(doc.id, stock)
        except Exception as e:
            print(f"❌ Error linking news to economic data for {stock}: {e}")
            continue

        # 3) Run in-place sentiment analysis on 'news' docs
        try:
            analyze_sentiment_and_store()
            print(f"🧠 Sentiment analysis completed for {stock}")
        except Exception as e:
            print(f"❌ Error running sentiment analysis for {stock}: {e}")
            continue

        # 4) Migrate leftover data from 'sentiment_analysis' => 'news'
        try:
            migrate_sentiment()
        except Exception as e:
            print(f"❌ Error migrating leftover sentiment docs: {e}")

        # 5) Fetch newly processed articles for this ticker
        try:
            related_news_query = db.collection("news").where("economic_data_id", "==", stock)
            related_news = list(related_news_query.stream())

            news_docs = []
            for doc_snap in related_news:
                # Only include docs with 'sentiment_label' & 'sentiment_score'
                if is_article_processed(doc_snap.id):
                    news_docs.append(doc_snap.to_dict())

            if not news_docs:
                print(f"📰 No *processed* articles for {stock}. Skipping recommendation.")
                continue

            print(f"📰 {len(news_docs)} processed articles ready for RAG analysis for {stock}")
        except Exception as e:
            print(f"❌ Error fetching related news for {stock}: {e}")
            continue

        # 6) Generate RAG Recommendation (two outputs)
        try:
            short_rec, detail_text = generate_rag_response(f"What's the outlook for {stock}?", news_docs)
            if "⚠️ No relevant data found" in detail_text:
                print(f"📊 Recommendation for {stock}: {detail_text}. Skipping storage.")
                continue

            print(f"📊 Recommendation for {stock}: {short_rec}")
        except Exception as e:
            print(f"❌ Error generating recommendation for {stock}: {e}")
            continue

        # 7) Evaluate & Store Recommendation
        try:
            print(f"📊 Evaluating recommendation for {stock}...")
            # Evaluate correctness using ONLY the short recommendation
            is_correct, latest_close, previous_close = evaluate_recommendation(stock, short_rec)
            print(f"✅ Evaluation for {stock}: Correct = {is_correct}, "
                  f"Latest = {latest_close}, Previous = {previous_close}")

            # Now store both short & detailed rec
            store_recommendation(
                stock_ticker=stock,
                short_rec=short_rec,          # e.g. "Buy"
                full_detail=detail_text,      # multi-paragraph explanation
                is_correct=is_correct,
                latest_close=latest_close,
                previous_close=previous_close
            )
        except Exception as e:
            print(f"❌ Error storing or evaluating recommendation for {stock}: {e}")

# ───────────────────────────────────────────────────────────────────────────────
# 🚀 Main Execution
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stock_tickers = ["TSLA", "AAPL", "MSFT", "NVDA", "NVO"]
    run_daily_pipeline(stock_tickers)
