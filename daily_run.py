import os
import openai
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials

# Agents
from Agents.news_agent import process_articles, link_news_to_economic_data, fetch_closing_prices, STOCK_MAPPING
from Agents.rag_agent import generate_rag_response, store_recommendation  
from Agents.sentiment_agent import analyze_sentiment_and_store, migrate_sentiment, is_article_processed

import warnings
import tensorflow as tf
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in environment variables.")

# ─────────────────────────────────────────────────────────────────────────────
# Suppress warnings
# ─────────────────────────────────────────────────────────────────────────────

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

# ─────────────────────────────────────────────────────────────────────────────
# Define experiment start date for tracking correctness
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_START_DATE = datetime(2025, 1, 18)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Firebase (if needed in this script)
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate Recommendation Correctness
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_recommendation(stock_ticker, recommendation):
    """
    Compare today's close vs. yesterday's close to see if the recommendation is correct.
    """
    latest_close, previous_close = fetch_closing_prices(stock_ticker)
    if not latest_close or not previous_close:
        print(f"⚠️ Missing stock price data for {stock_ticker}. Latest: {latest_close}, Previous: {previous_close}")
        return False, latest_close, previous_close

    price_movement = "up" if latest_close > previous_close else "down"
    is_correct = ((recommendation.lower() in ["buy", "hold"] and price_movement == "up") or
                  (recommendation.lower() == "sell" and price_movement == "down"))
    print(f"🔄 Stock Price for {stock_ticker}: Latest={latest_close}, Previous={previous_close}, Movement={price_movement}")
    print(f"🔍 Recommendation Correctness: {is_correct}")
    return is_correct, latest_close, previous_close

# ─────────────────────────────────────────────────────────────────────────────
# Store Recommendation Results
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
    """
    Store the final recommendations, sentiment summary, and evaluation results in Firestore.
    """
    try:
        experiment_day = (datetime.now() - EXPERIMENT_START_DATE).days
        experiment_day = min(experiment_day, 90)
        db.collection("model_recommendations").add({
            "stock_ticker": stock_ticker,
            "aggregator_recommendation": aggregator_recommendation,
            "gpt_recommendation": gpt_recommendation,
            "sentiment_summary": sentiment_summary,
            "recommendation_detail": full_text,
            "is_correct": is_correct,
            "latest_close": latest_close,
            "previous_close": previous_close,
            "timestamp": datetime.now().isoformat(),
            "experiment_day": experiment_day
        })
        print(f"✅ Stored recommendation for {stock_ticker}: GPT={gpt_recommendation}, Aggregator={aggregator_recommendation}, Correct={is_correct}")
    except Exception as e:
        print(f"❌ Error storing recommendation: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Check if a News Article Has Been Processed
# ─────────────────────────────────────────────────────────────────────────────

def is_article_processed(article_id):
    """
    Return True if the 'news' document with the given article_id has sentiment data.
    """
    try:
        doc_snapshot = db.collection("news").document(article_id).get()
        if not doc_snapshot.exists:
            return False
        data = doc_snapshot.to_dict()
        return ("sentiment_label" in data and "sentiment_score" in data)
    except Exception as e:
        print(f"❌ Error checking article {article_id}: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Run Daily Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_pipeline(stock_tickers, articles_per_stock=5):
    for stock in stock_tickers:
        print(f"\n🔍 Processing {stock}...")
        
        # 1) Fetch & store new articles
        process_articles([stock], articles_per_stock)
        
        # 2) Link articles to economic data
        newly_stored = db.collection("news").where("keywords", "array_contains", stock).stream()
        for doc in newly_stored:
            if not doc.to_dict().get("economic_data_id"):
                link_news_to_economic_data(doc.id, stock)
        
        # 3) Run sentiment analysis on the news articles
        analyze_sentiment_and_store()
        
        # 4) Migrate any leftover sentiment docs
        migrate_sentiment()  # This is now imported correctly from sentiment_agent
        
        # 5) Gather processed articles for this stock
        related_news_query = db.collection("news").where("economic_data_id", "==", stock)
        related_news = list(related_news_query.stream())
        news_docs = [doc_snap.to_dict() for doc_snap in related_news if is_article_processed(doc_snap.id)]
        if not news_docs:
            print(f"📰 No processed articles for {stock}. Skipping recommendation.")
            continue
        
        # 6) Generate recommendations using aggregated sentiment and GPT
        aggregator_rec, gpt_rec, sentiment_sum = generate_rag_response(f"What's the outlook for {stock}?", news_docs)
        print(f"Aggregator Recommendation: {aggregator_rec}")
        print(f"GPT Recommendation: {gpt_rec}")
        print(f"Sentiment Summary: {sentiment_sum}")
        
        # Optional: Compare aggregator vs. GPT recommendations
        if aggregator_rec == gpt_rec:
            print(f"✅ GPT agrees with aggregator: {gpt_rec}")
        else:
            print(f"🔀 GPT differs from aggregator. Aggregator={aggregator_rec}, GPT={gpt_rec}")
        
        # Optional: Evaluate bullish vs. bearish difference via simple mapping
        def rec_to_int(rec):
            return {"sell": -1, "hold": 0, "buy": 1}.get(rec.lower(), 0)
        agg_score = rec_to_int(aggregator_rec)
        gpt_score = rec_to_int(gpt_rec)
        if gpt_score > agg_score:
            print("GPT is more bullish than aggregator.")
        elif gpt_score < agg_score:
            print("GPT is more bearish than aggregator.")
        else:
            print("GPT has the same recommendation as aggregator.")
        
        # 7) Evaluate recommendation correctness vs. actual stock prices
        is_correct, latest_close, previous_close = evaluate_recommendation(stock, gpt_rec)
        
        # 8) Store the recommendation in Firestore
        store_recommendation(stock, aggregator_rec, gpt_rec, sentiment_sum, is_correct, latest_close, previous_close)
    
    print("\n✅ Daily Pipeline Workflow completed successfully!")

if __name__ == "__main__":
    stock_tickers = ["TSLA", "AAPL", "MSFT", "NVDA", "NVO"]
    run_daily_pipeline(stock_tickers)
