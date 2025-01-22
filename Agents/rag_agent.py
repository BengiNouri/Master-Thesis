import os
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf
import firebase_admin
from firebase_admin import firestore, credentials

# Load environment variables
load_dotenv()
# Verify that the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in environment variables")
else:
    print("OPENAI_API_KEY loaded successfully.")

# Use the new client instance approach
from openai import OpenAI
client = OpenAI(api_key=api_key)

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

db = initialize_firebase()

def load_stock_mapping():
    """
    Load mapping (e.g., company name to stock ticker) from Firestore.
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

def fetch_closing_prices(stock_ticker):
    """
    Fetch latest and previous closing prices for a stock using yfinance.
    """
    try:
        ticker = yf.Ticker(stock_ticker)
        hist = ticker.history(period="5d")
        if hist.empty or len(hist) < 2:
            print(f"⚠️ Not enough historical data for {stock_ticker}")
            return None, None
        latest_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        return latest_close, previous_close
    except Exception as e:
        print(f"❌ Error fetching closing prices for {stock_ticker}: {e}")
        return None, None

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

def generate_rag_response(query, documents):
    """
    Returns a tuple: (aggregator_rec, gpt_rec, sentiment_summary)
    """
    try:
        if not documents:
            print("⚠️ No relevant news docs found; defaulting to 'Hold'.")
            return ("Hold", "Hold", {"positive": 0.0, "neutral": 0.0, "negative": 0.0})

        sentiment_summary = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for doc in documents:
            label = doc.get("sentiment_label", "neutral").lower()
            score = float(doc.get("sentiment_score", 0.0))
            sentiment_summary[label] += score

        aggregator_rec = "Hold"
        if sentiment_summary["positive"] > sentiment_summary["negative"]:
            aggregator_rec = "Buy"
        elif sentiment_summary["negative"] > sentiment_summary["positive"]:
            aggregator_rec = "Sell"

        print(f"🔎 Aggregator suggests: {aggregator_rec}")
        print(f"🟢 Sentiment Summary: {sentiment_summary}")

        doc_lines = []
        for d in documents[:4]:
            title = d.get("title", "No Title")
            label = d.get("sentiment_label", "neutral").lower()
            score = d.get("sentiment_score", 0.0)
            doc_lines.append(f"- {title} ({label}, {score})")
        doc_context = "\n".join(doc_lines)

        user_prompt = (
            "You are a financial analyst. You MUST answer with exactly one word: Buy, Sell, or Hold.\n\n"
            f"Aggregator-based recommendation (from sentiment scores): {aggregator_rec}\n\n"
            f"Document Overviews:\n{doc_context}\n\n"
            "Based on the above, give your final single-word recommendation (Buy, Sell, or Hold). No additional words."
        )

        # Using the new client instance and a supported model
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Change to 'gpt-4' if you have access
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=5,
            temperature=0
        )

        gpt_raw = response.choices[0].message.content.strip()
        gpt_clean = gpt_raw.replace(".", "").strip().capitalize()
        if gpt_clean not in ["Buy", "Sell", "Hold"]:
            gpt_clean = "Hold"

        print(f"🤖 GPT Final Recommendation: {gpt_clean}")
        return (aggregator_rec, gpt_clean, sentiment_summary)

    except Exception as e:
        print(f"❌ Error in generate_rag_response: {e}")
        return ("Hold", "Hold", {"positive": 0.0, "neutral": 0.0, "negative": 0.0})

def store_recommendation(stock_ticker, aggregator_rec, gpt_rec, sentiment_sum):
    """
    Store the combined recommendations in Firestore under the 'recommendations' collection.
    """
    try:
        rec_doc = db.collection("recommendations").document(stock_ticker)
        rec_doc.set({
            "aggregator_rec": aggregator_rec,
            "gpt_rec": gpt_rec,
            "sentiment_sum": sentiment_sum,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"✅ Stored recommendation for {stock_ticker}: {aggregator_rec} | {gpt_rec}")
    except Exception as e:
        print(f"❌ Error storing recommendation for {stock_ticker}: {e}")

if __name__ == "__main__":
    # Example query for Tesla (TSLA)
    user_query = "What is the latest news about Tesla?"
    stock_ticker = STOCK_MAPPING.get("tesla", "TSLA")
    related_news = fetch_related_news(stock_ticker)
    
    # Unpack three values from generate_rag_response
    aggregator_rec, gpt_rec, sentiment_sum = generate_rag_response(user_query, related_news)

    print(f"Aggregator Recommendation: {aggregator_rec}")
    print(f"GPT Recommendation: {gpt_rec}")
    print(f"Sentiment Summary: {sentiment_sum}")

    # Compare the recommendations (optional)
    if aggregator_rec == gpt_rec:
        print(f"✅ GPT agrees with aggregator: {gpt_rec}")
    else:
        print(f"🔀 GPT differs from aggregator. Aggregator={aggregator_rec}, GPT={gpt_rec}")

    # Optional: Evaluate correctness vs. price movement
    latest_close, previous_close = fetch_closing_prices(stock_ticker)
    if latest_close and previous_close:
        is_correct = ((gpt_rec.lower() in ["buy", "hold"] and latest_close > previous_close) or
                      (gpt_rec.lower() == "sell" and latest_close < previous_close))
        print(f"📊 Recommendation correctness: {is_correct}")
        print(f"Today's close: {latest_close}, Yesterday's close: {previous_close}")
    else:
        print(f"⚠️ No stock price data available for {stock_ticker}.")

    # Store recommendation in Firestore
    store_recommendation(stock_ticker, aggregator_rec, gpt_rec, sentiment_sum)
