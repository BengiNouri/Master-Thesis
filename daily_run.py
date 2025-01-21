import os
import openai
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials

# Agents
from Agents.news_agent import process_articles
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
# ✅ Define experiment start date (for your correctness tracking)
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
    was "correct" for the movement. (Buy/Hold implies up; Sell implies down).
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

def store_recommendation(stock_ticker, recommendation, is_correct, latest_close, previous_close):
    """
    Store the recommendation results in Firestore. 
    Also track 'experiment_day' since a given start date.
    """
    try:
        experiment_day = (datetime.now() - EXPERIMENT_START_DATE).days
        experiment_day = min(experiment_day, 90)  # Cap at 90 days, as you indicated

        db.collection("model_recommendations").add({
            "stock_ticker": stock_ticker,
            "recommendation": recommendation,
            "is_correct": is_correct,
            "latest_close": latest_close,
            "previous_close": previous_close,
            "timestamp": datetime.now().isoformat(),
            "experiment_day": experiment_day
        })
        print(f"✅ Stored recommendation for {stock_ticker}: {recommendation} "
              f"| Correct: {is_correct} | Day: {experiment_day}")
    except Exception as e:
        print(f"❌ Error storing recommendation: {e}")

# ───────────────────────────────────────────────────────────────────────────────
# ✅ Check if Article Was Processed
# ───────────────────────────────────────────────────────────────────────────────

def is_article_processed(article_id):
    """
    Check if an article doc has already been processed for sentiment.
    We'll define "processed" as having 'sentiment_label' & 'sentiment_score'.
    """
    try:
        doc_snapshot = db.collection("news").document(article_id).get()
        if not doc_snapshot.exists:
            # Document doesn't even exist => not processed
            return False
        data = doc_snapshot.to_dict()
        # If the doc has these fields, we assume it's "processed".
        has_sentiment = ("sentiment_label" in data and "sentiment_score" in data)
        return has_sentiment
    except Exception as e:
        print(f"❌ Error checking if article {article_id} is processed: {e}")
        return False

# ───────────────────────────────────────────────────────────────────────────────
# 🔄 Run Daily Pipeline
# ───────────────────────────────────────────────────────────────────────────────

def run_daily_pipeline(stock_tickers, articles_per_stock=20):
    """
    1) Fetch & store new articles
    2) Analyze sentiment
    3) Fetch newly processed articles
    4) Generate RAG recommendation
    5) Evaluate correctness
    6) Store the recommendation
    """
    for stock in stock_tickers:
        print(f"\n🔍 Processing {stock}...")

        # Step 1: Fetch News Articles (store them in "news" collection)
        try:
            process_articles([stock], articles_per_stock)
            print(f"🔎 Fetched up to {articles_per_stock} articles for {stock}")
        except Exception as e:
            print(f"❌ Error processing articles for {stock}: {e}")
            continue

        # Step 2: Run Sentiment Analysis (updates each doc with sentiment_label/score)
        try:
            analyze_sentiment_and_store()
            print(f"🧠 Sentiment analysis completed for {stock}")
        except Exception as e:
            print(f"❌ Error running sentiment analysis for {stock}: {e}")
            continue

        # Step 3: Fetch related news (docs that link to this stock)
        #         Keep only those that are now "processed" with sentiment
        try:
            related_news = db.collection("news") \
                .where("economic_data_id", "==", stock) \
                .stream()

            # Only pick docs that have sentiment fields (i.e., "processed").
            # If you want the opposite (unprocessed), invert the check.
            news_docs = []
            for doc in related_news:
                if is_article_processed(doc.id):
                    news_docs.append(doc.to_dict())

            if not news_docs:
                print(f"📰 No *processed* articles for {stock}. Skipping recommendation.")
                continue

            print(f"📰 {len(news_docs)} processed articles ready for RAG analysis for {stock}")
        except Exception as e:
            print(f"❌ Error fetching related news for {stock}: {e}")
            continue

        # Step 4: Generate Recommendation
        try:
            recommendation = generate_rag_response(f"What's the outlook for {stock}?", news_docs)

            if "⚠️ No relevant data found" in recommendation:
                print(f"📊 Recommendation for {stock}: {recommendation}. Skipping storage.")
                continue

            print(f"📊 Recommendation for {stock}: {recommendation}")
        except Exception as e:
            print(f"❌ Error generating recommendation for {stock}: {e}")
            continue

        # Step 5: Evaluate Recommendation
        try:
            print(f"📊 Evaluating recommendation for {stock}...")
            is_correct, latest_close, previous_close = evaluate_recommendation(stock, recommendation)
            print(f"✅ Evaluation for {stock}: Correct = {is_correct}, "
                  f"Latest = {latest_close}, Previous = {previous_close}")
        except Exception as e:
            print(f"❌ Error evaluating recommendation for {stock}: {e}")
            continue

        # Step 6: Store Recommendation
        try:
            store_recommendation(stock, recommendation, is_correct, latest_close, previous_close)
        except Exception as e:
            print(f"❌ Error storing recommendation for {stock}: {e}")

# ───────────────────────────────────────────────────────────────────────────────
# 🚀 Main Execution
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stock_tickers = ["TSLA", "AAPL", "MSFT", "NVDA", "NVO"]
    run_daily_pipeline(stock_tickers)
