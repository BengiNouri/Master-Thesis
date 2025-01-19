import os
import sys
import requests
from datetime import datetime
import yfinance as yf
import firebase_admin
from firebase_admin import firestore, credentials
from dotenv import load_dotenv

# TensorFlow and Torch configuration to suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import sentiment analysis
from Agents.sentiment_agent import analyze_sentiment_and_store

# 🔐 Initialize Firebase
def initialize_firebase():
    """
    Initialize Firebase with fallback paths.
    """
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    if not firebase_admin._apps:
        cred_path = primary_path if os.path.exists(primary_path) else fallback_path
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    return firestore.client()

# Initialize Firebase
db = initialize_firebase()

# ✅ Build Stock Mapping
def build_stock_mapping():
    """
    Build mapping between company names and stock tickers from Firestore.
    """
    stock_mapping = {}
    try:
        docs = db.collection("latest_economic_data").stream()
        for doc in docs:
            data = doc.to_dict()
            company_name = data.get("long_name")
            stock_ticker = data.get("stock_ticker")

            if company_name and stock_ticker:
                stock_mapping[company_name.lower().strip()] = stock_ticker.upper().strip()
                stock_mapping[stock_ticker.lower().strip()] = stock_ticker.upper().strip()
            else:
                print(f"⚠️ Skipped invalid or incomplete data: {data}")

        print("✅ Stock mapping loaded successfully from Firestore.")
    except Exception as e:
        print(f"❌ Error loading stock mapping: {e}")
    return stock_mapping

STOCK_MAPPING = build_stock_mapping()

# ✅ Fetch Stock Prices
def fetch_closing_prices(stock_ticker):
    """
    Fetch the latest and previous closing prices for a stock.
    """
    try:
        ticker = yf.Ticker(stock_ticker)
        hist = ticker.history(period="5d")
        latest_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        return latest_close, previous_close
    except Exception as e:
        print(f"❌ Error fetching closing prices for {stock_ticker}: {e}")
        return None, None

# ✅ Fetch News Articles
def fetch_news_articles(keywords, page_size=10):
    """
    Fetch news articles using NewsAPI.
    """
    try:
        if not NEWS_API_KEY:
            raise ValueError("NEWS_API_KEY is not set.")
        
        query = " OR ".join(keywords)
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size
        }
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except Exception as e:
        print(f"❌ Error fetching news articles: {e}")
        return []

# ✅ Store Data in Firestore
def store_in_firebase(collection_name, data):
    """
    Store documents in the specified Firestore collection.
    """
    try:
        for item in data:
            doc_ref = db.collection(collection_name).document()
            item["doc_id"] = doc_ref.id
            doc_ref.set(item)
        print(f"✅ Stored {len(data)} documents in '{collection_name}'.")
    except Exception as e:
        print(f"❌ Error storing data: {e}")

# ✅ Link News to Economic Data
def link_news_to_economic_data(news_id, keyword):
    """
    Link news articles to economic data.
    """
    try:
        stock_ticker = STOCK_MAPPING.get(keyword.lower())
        if not stock_ticker:
            print(f"⚠️ No stock ticker found for '{keyword}'.")
            return

        db.collection("latest_economic_data").document(stock_ticker).update({
            "linked_news_ids": firestore.ArrayUnion([news_id])
        })
        db.collection("news").document(news_id).update({
            "economic_data_id": stock_ticker
        })
        print(f"🔗 Linked news ID {news_id} to {stock_ticker}")
    except Exception as e:
        print(f"❌ Error linking news to economic data: {e}")

# ✅ Evaluate Recommendations
def evaluate_recommendation(stock_ticker, recommendation):
    """
    Evaluate if the recommendation was correct.
    """
    latest_close, previous_close = fetch_closing_prices(stock_ticker)
    if not latest_close or not previous_close:
        return False, latest_close, previous_close

    price_movement = "up" if latest_close > previous_close else "down"
    is_correct = ((recommendation.lower() in ["buy", "hold"] and price_movement == "up") or
                  (recommendation.lower() == "sell" and price_movement == "down"))
    return is_correct, latest_close, previous_close

# ✅ Store Recommendation Results
def store_recommendation(stock_ticker, recommendation, is_correct, latest_close, previous_close):
    """
    Store model recommendations in Firestore.
    """
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

# ✅ Process Articles Workflow
def process_articles(keywords, page_size=10):
    """
    Fetch, store, analyze, and link news articles.
    """
    print(f"🔎 Fetching news articles for: {keywords}")
    articles = fetch_news_articles(keywords, page_size)

    if not articles:
        print("⚠️ No articles found.")
        return

    structured_articles = [
        {
            "title": article.get("title"),
            "content": article.get("content"),
            "url": article.get("url"),
            "publishedAt": article.get("publishedAt"),
            "source": article.get("source", {}).get("name"),
            "keywords": keywords,
            "sentiment_id": None,
            "economic_data_id": None
        }
        for article in articles
    ]

    store_in_firebase("news", structured_articles)
    analyze_sentiment_and_store()

    for article in structured_articles:
        for keyword in keywords:
            link_news_to_economic_data(article["doc_id"], keyword)

    for keyword in keywords:
        recommendation = "buy"  # Placeholder recommendation
        is_correct, latest_close, previous_close = evaluate_recommendation(STOCK_MAPPING[keyword.lower()], recommendation)
        store_recommendation(STOCK_MAPPING[keyword.lower()], recommendation, is_correct, latest_close, previous_close)

    print("✅ Workflow completed successfully!")

# 🚀 Main Execution
if __name__ == "__main__":
    keywords = ["Tesla", "TSLA"]
    page_size = 5
    process_articles(keywords, page_size)
