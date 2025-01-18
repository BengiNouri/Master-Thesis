import os
import openai
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials
from Agents.news_agent import process_articles
from Agents.rag_agent import generate_rag_response
from Agents.sentiment_agent import analyze_sentiment_and_store
from fuzzywuzzy import process

# ✅ Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🔐 Initialize Firebase
def initialize_firebase():
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    if not firebase_admin._apps:
        cred_path = primary_path if os.path.exists(primary_path) else fallback_path
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    return firestore.client()

# Initialize Firebase
db = initialize_firebase()

# 📈 Fetch Stock Prices
def fetch_closing_prices(stock_ticker):
    try:
        ticker = yf.Ticker(stock_ticker)
        hist = ticker.history(period="5d")
        latest_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        return latest_close, previous_close
    except Exception as e:
        print(f"❌ Error fetching stock data for {stock_ticker}: {e}")
        return None, None

# 📊 Evaluate Recommendation
def evaluate_recommendation(stock_ticker, recommendation):
    latest_close, previous_close = fetch_closing_prices(stock_ticker)
    if not latest_close or not previous_close:
        return False, latest_close, previous_close

    price_movement = "up" if latest_close > previous_close else "down"
    is_correct = ((recommendation.lower() in ["buy", "hold"] and price_movement == "up") or
                  (recommendation.lower() == "sell" and price_movement == "down"))
    return is_correct, latest_close, previous_close

# 📝 Store Recommendation Results
def store_recommendation(stock_ticker, recommendation, is_correct, latest_close, previous_close):
    try:
        db.collection("model_recommendations").add({
            "stock_ticker": stock_ticker,
            "recommendation": recommendation,
            "is_correct": is_correct,
            "latest_close": latest_close,
            "previous_close": previous_close,
            "timestamp": datetime.now().isoformat()
        })
        print(f"✅ Stored recommendation for {stock_ticker}: {recommendation} | Correct: {is_correct}")
    except Exception as e:
        print(f"❌ Error storing recommendation: {e}")

# 🔄 Run Daily Pipeline
def run_daily_pipeline(stock_tickers, articles_per_stock=20):
    for stock in stock_tickers:
        print(f"\n🔍 Processing {stock}...")

        # Step 1: Fetch News Articles
        process_articles([stock], articles_per_stock)

        # Step 2: Run Sentiment Analysis
        analyze_sentiment_and_store()

        # Step 3: Fetch Related News
        related_news = db.collection("news").where("economic_data_id", "==", stock).stream()
        news_docs = [doc.to_dict() for doc in related_news]

        # Step 4: Generate Recommendation
        recommendation = generate_rag_response(f"What's the outlook for {stock}?", news_docs)

        # Step 5: Evaluate Recommendation
        is_correct, latest_close, previous_close = evaluate_recommendation(stock, recommendation)

        # Step 6: Store Recommendation Result
        store_recommendation(stock, recommendation, is_correct, latest_close, previous_close)

# 🚀 Main Execution
if __name__ == "__main__":
    stock_tickers = ["TSLA", "AAPL", "MSFT", "NVDA", "NVO"]  # Stocks to analyze
    run_daily_pipeline(stock_tickers)
