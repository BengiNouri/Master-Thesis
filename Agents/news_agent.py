import os
import sys
import requests
import yfinance as yf
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import firestore, initialize_app, credentials
from dotenv import load_dotenv

# TensorFlow workaround
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import agents
from Agents.sentiment_agent import analyze_sentiment_and_store

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

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

# 🔥 Dynamic Mapping between company names and stock tickers
def build_stock_mapping():
    stock_mapping = {}
    try:
        docs = db.collection("latest_economic_data").stream()
        for doc in docs:
            data = doc.to_dict()
            company_name = data.get("long_name", "").lower()
            stock_ticker = data.get("stock_ticker", "").upper()
            if company_name and stock_ticker:
                stock_mapping[company_name] = stock_ticker
                stock_mapping[stock_ticker.lower()] = stock_ticker
        print("✅ Stock mapping loaded from Firestore.")
    except Exception as e:
        print(f"❌ Error building stock mapping: {e}")
    return stock_mapping

STOCK_MAPPING = build_stock_mapping()

# 📈 Fetch Stock Prices
def fetch_closing_prices(stock_ticker):
    try:
        ticker = yf.Ticker(stock_ticker)
        hist = ticker.history(period="5d")
        latest_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        return latest_close, previous_close
    except Exception as e:
        print(f"❌ Error fetching closing prices for {stock_ticker}: {e}")
        return None, None

# 📰 Fetch News Articles
def fetch_news_articles(keywords, page_size=10):
    try:
        if not NEWS_API_KEY:
            raise ValueError("NEWS_API_KEY is not set.")
        query = " OR ".join(keywords)
        params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt", "pageSize": page_size}
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return []

# 🗂️ Store Data in Firestore
def store_in_firebase(collection_name, data):
    try:
        for item in data:
            doc_ref = db.collection(collection_name).document()
            item["doc_id"] = doc_ref.id
            doc_ref.set(item)
        print(f"✅ Stored {len(data)} documents in '{collection_name}'.")
    except Exception as e:
        print(f"❌ Error storing data: {e}")

def link_news_to_economic_data(news_id, keyword):
    """
    Link news articles to economic data based on the keyword.
    """
    try:
        stock_ticker = STOCK_MAPPING.get(keyword.lower())

        if not stock_ticker:
            print(f"⚠️ No stock ticker found for '{keyword}'.")
            return

        # Link the news article to the economic data
        db.collection("news").document(news_id).update({
            "economic_data_id": stock_ticker
        })

        # Link back in economic data
        db.collection("latest_economic_data").document(stock_ticker).update({
            "linked_news_ids": firestore.ArrayUnion([news_id])
        })

        print(f"🔗 Linked news ID {news_id} to economic data ID {stock_ticker}")

    except Exception as e:
        print(f"❌ Error linking news: {e}")


# 📊 Evaluate Recommendations
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

# 🔎 Full Workflow
def process_articles(keywords, page_size=10):
    print(f"🔎 Fetching news articles for: {keywords}")
    articles = fetch_news_articles(keywords, page_size)
    if not articles:
        print(f"⚠️ No articles found.")
        return

    structured_articles = [
        {"title": article.get("title"), "content": article.get("content"),
         "url": article.get("url"), "publishedAt": article.get("publishedAt"),
         "source": article.get("source", {}).get("name"),
         "keywords": keywords, "sentiment_id": None, "economic_data_id": None}
        for article in articles
    ]
    store_in_firebase("news", structured_articles)
    print("🧠 Triggering sentiment analysis...")
    analyze_sentiment_and_store()
    
    print("🔗 Linking news articles...")
    for article in structured_articles:
        for keyword in keywords:
            link_news_to_economic_data(article["doc_id"], keyword)

    for keyword in keywords:
        recommendation = "buy"  # Placeholder
        is_correct, latest_close, previous_close = evaluate_recommendation(STOCK_MAPPING[keyword.lower()], recommendation)
        store_recommendation(STOCK_MAPPING[keyword.lower()], recommendation, is_correct, latest_close, previous_close)

    print("✅ Workflow completed successfully!")

# 🚀 Main Execution
if __name__ == "__main__":
    keywords = ["Tesla", "TSLA"]
    page_size = 5
    process_articles(keywords, page_size)
