import os
import openai
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials

# ✅ Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not openai.api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in environment variables.")

# 🔐 Initialize Firebase
def initialize_firebase():
    """
    Initialize Firebase and return a Firestore client.
    """
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    if not firebase_admin._apps:
        cred_path = primary_path if os.path.exists(primary_path) else fallback_path
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"❌ Firebase credentials not found in {cred_path}")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    return firestore.client()

# Initialize Firestore
db = initialize_firebase()

# 🔍 Load Stock Mapping Dynamically
def load_stock_mapping():
    """
    Load stock mapping (e.g., company name to stock ticker) from Firestore.
    """
    mapping = {}
    try:
        docs = db.collection("latest_economic_data").stream()
        for doc in docs:
            data = doc.to_dict()
            stock_ticker = doc.id.upper().strip() if doc.id else None
            company_name = data.get("long_name", "").lower().strip() if data.get("long_name") else None
            if stock_ticker:
                mapping[stock_ticker] = stock_ticker
            if company_name:
                mapping[company_name] = stock_ticker
        print("✅ Stock mapping loaded from Firestore.")
    except Exception as e:
        print(f"⚠️ Error loading stock mapping: {e}")
    return mapping

STOCK_MAPPING = load_stock_mapping()

# 📈 Fetch Stock Prices
def fetch_closing_prices(stock_ticker):
    """
    Fetch the latest and previous closing prices for a given stock ticker.
    """
    try:
        ticker = yf.Ticker(stock_ticker)
        hist = ticker.history(period="5d")
        if hist.empty or len(hist) < 2:
            print(f"⚠️ Not enough historical data for {stock_ticker}")
            return None, None
        latest_close, previous_close = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
        return latest_close, previous_close
    except Exception as e:
        print(f"❌ Error fetching stock data for {stock_ticker}: {e}")
        return None, None

# 📰 Fetch Related News
def fetch_related_news(stock_ticker):
    """
    Fetch related news articles for a given stock ticker from Firestore.
    """
    try:
        econ_doc = db.collection("latest_economic_data").document(stock_ticker).get()
        if not econ_doc.exists:
            print(f"⚠️ No economic data found for {stock_ticker}.")
            return []

        linked_news_ids = econ_doc.to_dict().get("linked_news_ids", [])
        news_articles = []
        for news_id in linked_news_ids:
            news_doc = db.collection("news").document(news_id).get()
            if news_doc.exists:
                news_articles.append(news_doc.to_dict())
        return news_articles
    except Exception as e:
        print(f"❌ Error fetching related news: {e}")
        return []

import openai

import openai

import openai

def generate_rag_response(query, documents):
    """
    Generate a financial recommendation based on sentiment trends and context.
    """
    try:
        if not documents:
            return "⚠️ No relevant data found for your query."

        # Aggregate sentiment trends
        sentiment_summary = {"positive": 0, "neutral": 0, "negative": 0}
        for doc in documents:
            sentiment_label = doc.get("sentiment_label", "neutral").lower()
            sentiment_score = doc.get("sentiment_score", 0)
            sentiment_summary[sentiment_label] += sentiment_score
            print(f"📰 Title: {doc.get('title', 'No Title')}, Sentiment: {sentiment_label} ({sentiment_score})")

        print(f"🟢 Sentiment Summary for Query: {sentiment_summary}")

        # Derive recommendation based on sentiment trends
        if sentiment_summary["positive"] > sentiment_summary["negative"]:
            recommendation = "Buy"
        elif sentiment_summary["negative"] > sentiment_summary["positive"]:
            recommendation = "Sell"
        else:
            recommendation = "Hold"

        # Formulate context for OpenAI prompt
        context = "\n\n".join([
            f"📰 **Title:** {doc.get('title', 'No Title')}\n"
            f"📄 **Content:** {doc.get('content', 'No Content')}\n"
            f"🟢 **Sentiment:** {doc.get('sentiment_label', 'No Sentiment')} "
            f"({doc.get('sentiment_score', 'N/A')})"
            for doc in documents if doc
        ])

        # OpenAI API call with updated syntax
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": (
                    f"Analyze the following financial news and sentiment data:\n\n"
                    f"{context}\n\n"
                    f"Based on the above, what is your recommendation for {query}? "
                    f"(Buy, Hold, or Sell)"
                )},
            ],
            max_tokens=500,
            temperature=0.3,
        )

        # Extract and return recommendation
        recommendation_output = response.choices[0].message.content.strip()
        print(f"🤖 Model Recommendation: {recommendation_output}")
        return recommendation_output

    except Exception as e:
        print(f"❌ Error generating RAG response: {e}")
        return "⚠️ An error occurred. Please try again later."

    except Exception as e:
        print(f"❌ Error generating RAG response: {e}")
        return "⚠️ An error occurred. Please try again later."

# 🚀 Main Execution
if __name__ == "__main__":
    user_query = "What is the latest news about Tesla?"
    stock_ticker = STOCK_MAPPING.get("tesla", "TSLA")
    related_news = fetch_related_news(stock_ticker)
    recommendation = generate_rag_response(user_query, related_news)

    latest_close, previous_close = fetch_closing_prices(stock_ticker)
    if latest_close and previous_close:
        is_correct = latest_close > previous_close
        print(f"📊 Recommendation: {recommendation} | Correct: {is_correct}")
    else:
        print(f"⚠️ No stock price data available for {stock_ticker}.")
