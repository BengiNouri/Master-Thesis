import os
import openai
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials
from fuzzywuzzy import process

# ✅ Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in environment variables.")

# 🔐 Initialize Firebase
def initialize_firebase():
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
    mapping = {}
    try:
        docs = db.collection("latest_economic_data").stream()
        for doc in docs:
            data = doc.to_dict()
            stock_ticker = doc.id.upper().strip() if doc.id else None
            company_name = data.get("long_name", "").lower().strip()
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

# ✅ Define experiment start date
EXPERIMENT_START_DATE = datetime(2025, 1, 18)

# 📝 Store Recommendation Results with experiment_day
def store_recommendation(stock_ticker, recommendation, is_correct, latest_close, previous_close):
    try:
        today = datetime.now()
        experiment_day = max(0, min((today - EXPERIMENT_START_DATE).days, 90))  # Ensure day is 0-90

        db.collection("model_recommendations").add({
            "stock_ticker": stock_ticker,
            "recommendation": recommendation,
            "is_correct": is_correct,
            "latest_close": latest_close,
            "previous_close": previous_close,
            "timestamp": today.isoformat(),
            "experiment_day": experiment_day
        })
        print(f"✅ Stored recommendation for {stock_ticker}: {recommendation} | Correct: {is_correct} | Day: {experiment_day}")

    except Exception as e:
        print(f"❌ Error storing recommendation: {e}")

# 🧠 Generate Financial Recommendation
def generate_rag_response(query, documents):
    try:
        if not documents:
            return "⚠️ No relevant data found for your query."

        context = "\n\n".join([
            f"📰 **Title:** {doc.get('title', 'No Title')}\n"
            f"📄 **Summary:** {doc.get('summary', 'No Summary')}\n"
            f"📄 **Content:** {doc.get('content', 'No Content')}\n"
            f"🟢 **Sentiment:** {doc.get('sentiment', {}).get('label', 'No Sentiment')} "
            f"({doc.get('sentiment', {}).get('score', 'N/A')})"
            for doc in documents if doc
        ])

        prompt = (
            f"You are a financial analyst. Analyze the following financial news and sentiment analysis to give a direct investment recommendation: **Buy**, **Hold**, or **Sell**.\n\n"
            f"Consider recent news, sentiment trends, and market movements. Be decisive and explain the reasoning behind your recommendation.\n\n"
            f"{context}\n\n"
            f"❓ **Question:** {query}\n"
            f"💡 **Answer (Buy, Hold, or Sell):**"
        )

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use GPT-4 for more advanced responses
            messages=[
                {"role": "system", "content": "You are a highly skilled financial advisor."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.3
        )

        recommendation = response.choices[0].message.content.strip()
        print(f"🤖 Model Recommendation: {recommendation}")
        return recommendation

    except Exception as e:
        print(f"❌ Error generating RAG response: {e}")
        return "⚠️ An error occurred. Please try again later."

# 🚀 Main Execution
if __name__ == "__main__":
    user_query = "What is the latest news about Microsoft?"
    stock_ticker = STOCK_MAPPING.get("microsoft", "MSFT")
    print(f"🔍 Query: {user_query} for ticker: {stock_ticker}")

    related_news = fetch_related_news(stock_ticker)
    recommendation = generate_rag_response(user_query, related_news)
    latest_close, previous_close = fetch_closing_prices(stock_ticker)

    if latest_close and previous_close:
        is_correct = latest_close > previous_close
        store_recommendation(stock_ticker, recommendation, is_correct, latest_close, previous_close)
        print(f"📊 Evaluation complete for {stock_ticker}: Correct = {is_correct}")
    else:
        print(f"⚠️ Skipping evaluation due to missing price data for {stock_ticker}.")
