import os
import openai
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials
from fuzzywuzzy import process  # Install via: pip install fuzzywuzzy python-Levenshtein

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

# 🔍 Load Stock Mapping Dynamically
def load_stock_mapping():
    mapping = {}
    try:
        docs = db.collection("latest_economic_data").stream()
        for doc in docs:
            data = doc.to_dict()
            stock_ticker = doc.id.upper().strip()
            company_name = data.get("long_name", "").lower().strip()
            if stock_ticker:
                mapping[stock_ticker] = stock_ticker
            if company_name:
                mapping[company_name] = stock_ticker
                mapping[company_name.replace("corporation", "").strip()] = stock_ticker
                mapping[company_name.replace("inc.", "").strip()] = stock_ticker
        print("✅ Dynamic stock mapping loaded from Firestore.")
    except Exception as e:
        print(f"⚠️ Error loading stock mapping: {e}")
    return mapping

# Load stock mapping globally
STOCK_MAPPING = load_stock_mapping()

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

# 🔍 Improved Fuzzy Matching for Keywords
def find_best_match(keyword):
    keyword = keyword.lower().strip()
    if keyword in STOCK_MAPPING:
        return STOCK_MAPPING[keyword]

    all_keys = list(STOCK_MAPPING.keys())
    best_match, score = process.extractOne(keyword, all_keys)

    if score > 75:
        return STOCK_MAPPING[best_match]
    return None

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

# 🤖 Generate RAG Response with Summaries and Sentiments
def generate_rag_response(query, documents):
    try:
        if not documents:
            return "⚠️ No relevant data found for your query."

        # Construct context from related news including summaries
        context = "\n\n".join([
            f"📰 Title: {doc.get('title', 'No Title')}\n"
            f"📄 Summary: {doc.get('summary', 'No Summary')}\n"
            f"📄 Content: {doc.get('content', 'No Content')}\n"
            f"🟢 Sentiment: {doc.get('sentiment_id', 'No Sentiment')}"
            for doc in documents
        ])

        prompt = (
            f"Analyze the following financial news context and provide a clear investment recommendation "
            f"(Buy, Hold, or Sell) with reasoning. Use the summarized insights and sentiment analysis for accuracy. "
            f"Include relevant market data and trends to support your advice.\n\n"
            f"{context}\n\n"
            f"❓ Question: {query}\n"
            f"💡 Answer:"
        )

        # ✅ OpenAI API Call
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7
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

    # Dynamically find the best stock ticker
    stock_ticker = find_best_match(user_query) or "MSFT"
    print(f"🔍 Query: {user_query} for ticker: {stock_ticker}")

    # Fetch related news
    related_news = fetch_related_news(stock_ticker)

    # Generate recommendation
    recommendation = generate_rag_response(user_query, related_news)

    # Evaluate the recommendation
    is_correct, latest_close, previous_close = evaluate_recommendation(stock_ticker, recommendation)

    # Store the result in Firestore
    store_recommendation(stock_ticker, recommendation, is_correct, latest_close, previous_close)

    print(f"📊 Evaluation complete for {stock_ticker}: Correct = {is_correct}")
