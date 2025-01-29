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
# Configurations
# ─────────────────────────────────────────────────────────────────────────────
VERBOSE = True  # Set to False to reduce console output

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
EXPERIMENT_START_DATE = datetime(2025, 1, 22)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Firebase
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
    Compare today's close vs. yesterday's close to assess if the recommendation is correct.
    (Buy/Hold implies price should be up; Sell implies price should be down.)
    """
    latest_close, previous_close = fetch_closing_prices(stock_ticker)
    if not latest_close or not previous_close:
        if VERBOSE:
            print(f"⚠️ Missing stock price data for {stock_ticker}. Latest: {latest_close}, Previous: {previous_close}")
        return False, latest_close, previous_close

    movement = "up" if latest_close > previous_close else "down"
    is_correct = ((recommendation.lower() in ["buy", "hold"] and movement == "up") or
                  (recommendation.lower() == "sell" and movement == "down"))
    if VERBOSE:
        print(f"Stock: {stock_ticker} | Latest: {latest_close} | Previous: {previous_close} | Movement: {movement}")
        print(f"Recommendation: {recommendation} -> Correct: {is_correct}")
    return is_correct, latest_close, previous_close

# ─────────────────────────────────────────────────────────────────────────────
# Store Recommendation Results (Optimized Output Format)
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
    Store final recommendations, sentiment summary, and evaluation results in Firestore.
    Also stores economic data separately in the 'economic_data' collection.
    """
    try:
        experiment_day = (datetime.now() - EXPERIMENT_START_DATE).days
        experiment_day = min(experiment_day, 90)

        # Calculate price change percentage
        price_change = ((latest_close - previous_close) / previous_close) * 100 if previous_close else 0

        # Store in model_recommendations collection
        recommendation_ref = db.collection("model_recommendations").add({
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

        # Store economic data separately
        economic_data_ref = db.collection("economic_data").add({
            "stock_ticker": stock_ticker,
            "experiment_day": experiment_day,
            "latest_close": latest_close,
            "previous_close": previous_close,
            "price_change": price_change,
            "sentiment_summary": sentiment_summary,
            "timestamp": datetime.now().isoformat(),
            "recommendation_id": recommendation_ref[1].id  # Link economic data to recommendation
        })

        # Optimized Console Output
        print(f"""
⚠️ No leftover sentiment docs found, or all had missing news_id.

🔎 **Aggregator Suggests:** {aggregator_recommendation}  
🤖 **GPT Recommendation:** {gpt_recommendation} ({"✅ Aligned" if aggregator_recommendation == gpt_recommendation else "⚠️ Different"})  

📊 **Sentiment Scores:**
   - **Positive:** {sentiment_summary.get('positive', 0):.2f}%
   - **Neutral:** {sentiment_summary.get('neutral', 0):.2f}%
   - **Negative:** {sentiment_summary.get('negative', 0):.2f}%

📈 **Stock: {stock_ticker}**
   - **Latest Price:** {latest_close:.2f} ({'⬆' if latest_close > previous_close else '⬇'} {price_change:.2f}%)
   - **Previous Price:** {previous_close:.2f}
   - **{gpt_recommendation.capitalize()} Recommendation** {"✅ Correct" if is_correct else "❌ Incorrect"}

🗄️ **Data Stored:**
   - **Model Recommendations:** ID {recommendation_ref[1].id}
   - **Economic Data:** ID {economic_data_ref[1].id}

============================================================
✅ Stored Data for {stock_ticker}: GPT={gpt_recommendation}, Aggregator={aggregator_recommendation}, Correct={is_correct}
============================================================
""")

    except Exception as e:
        print(f"❌ Error storing recommendation for {stock_ticker}: {e}")



# ─────────────────────────────────────────────────────────────────────────────
# Check if a News Article Has Been Processed
# ─────────────────────────────────────────────────────────────────────────────
def is_article_processed(article_id):
    """
    Return True if the news document with given article_id contains sentiment data.
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

def run_daily_pipeline(stock_tickers, articles_per_stock=20):
    for stock in stock_tickers:
        print("\n" + "="*60)
        print(f"Processing Stock: {stock}")
        print("="*60)
        
        # 1) Fetch & store new articles
        print("\n[News Fetch]")
        process_articles([stock], articles_per_stock)
        print(f"Fetched & stored articles for: [{stock}]")
        
        # 2) Link articles to economic data
        newly_stored = db.collection("news").where("keywords", "array_contains", stock).stream()
        for doc in newly_stored:
            if not doc.to_dict().get("economic_data_id"):
                link_news_to_economic_data(doc.id, stock)
        
        # 3) Run sentiment analysis and migrate leftovers
        print("[Sentiment Analysis]")
        analyze_sentiment_and_store()
        migrate_sentiment()
        
        # 4) Gather only processed articles for this stock
        related_news = db.collection("news").where("economic_data_id", "==", stock).stream()
        news_docs = [doc.to_dict() for doc in related_news if "sentiment_label" in doc.to_dict()]
        
        if not news_docs:
            print(f"⚠️ No processed articles for {stock}. Skipping recommendation.")
            continue
        
        # 5) Generate recommendations
        aggregator_rec, gpt_rec, sentiment_sum = generate_rag_response(f"What's the outlook for {stock}?", news_docs)
        
        # 6) Evaluate correctness vs. stock prices
        is_correct, latest_close, previous_close = evaluate_recommendation(stock, gpt_rec)

        # 7) Store final recommendation (Handles all logging)
        store_recommendation(stock, aggregator_rec, gpt_rec, sentiment_sum, is_correct, latest_close, previous_close)
    
    print("\n✅ Daily Pipeline Workflow completed successfully!")


if __name__ == "__main__":
    stock_tickers = ["TSLA", "AAPL", "MSFT", "NVDA", "NVO"]
    run_daily_pipeline(stock_tickers)
