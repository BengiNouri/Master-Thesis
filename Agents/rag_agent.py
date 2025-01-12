import os
import openai
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials

# Load environment variables
load_dotenv()

# Initialize OpenAI API (Updated for v1.0.0+)
openai.api_key = os.getenv("OPENAI_API_KEY")

import firebase_admin
from firebase_admin import firestore, credentials

import os
import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    """
    Initialize Firebase with a fallback if the primary path fails.
    """
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        # Try the primary path first
        if os.path.exists(primary_path):
            cred = credentials.Certificate(primary_path)
        # If not, use the fallback path
        elif os.path.exists(fallback_path):
            cred = credentials.Certificate(fallback_path)
        else:
            raise FileNotFoundError("Firebase credentials file not found in both paths.")
        
        firebase_admin.initialize_app(cred)

    return firestore.client()

# Initialize Firebase
db = initialize_firebase()


def load_stock_mapping():
    """
    Dynamically load stock mappings from Firestore.
    """
    mapping = {}
    try:
        docs = db.collection("latest_economic_data").stream()
        for doc in docs:
            data = doc.to_dict()
            company_name = data.get("long_name", "").lower().strip()
            stock_ticker = doc.id.upper().strip()
            mapping[company_name] = stock_ticker
        print("✅ Stock mapping loaded from Firestore.")
    except Exception as e:
        print(f"⚠️ Error loading stock mapping: {e}")
    return mapping

# Load stock mapping globally
STOCK_MAPPING = load_stock_mapping()

def fetch_related_news(stock_ticker):
    """
    Fetch related news articles for the given stock ticker.
    """
    try:
        econ_doc = db.collection("latest_economic_data").document(stock_ticker).get()
        if not econ_doc.exists:
            print(f"⚠️ No economic data found for {stock_ticker}.")
            return []

        linked_news_ids = econ_doc.to_dict().get("linked_news_ids", [])
        if not linked_news_ids:
            print(f"⚠️ No linked news articles for {stock_ticker}.")
            return []

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
    Generate a RAG response using OpenAI API (updated for v1.0.0+).
    """
    try:
        if not documents:
            return "⚠️ No relevant data found for your query."

        # Construct context from related news
        context = "\n\n".join([
            f"📰 Title: {doc.get('title', 'No Title')}\n📄 Content: {doc.get('content', 'No Content')}"
            for doc in documents
        ])

        prompt = (
               f"Analyze the following context and provide a clear investment recommendation (Buy, Hold, or Sell) "
               f"with reasoning. Include relevant market data and trends to support your advice, but remind the user to do their own research.\n\n"
               f"{context}\n\n"
               f"❓ Question: {query}\n"
               f"💡 Answer:"
        )

        # ✅ Updated OpenAI API call for v1.0.0+
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Use the GPT-4 4B model
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error generating RAG response: {e}")
        return "⚠️ An error occurred. Please try again later."

if __name__ == "__main__":
    user_query = "What is the latest news about Tesla?"
    stock_ticker = STOCK_MAPPING.get("tesla", "TSLA")  # Default to TSLA if mapping fails

    print(f"🔍 Query: {user_query}")

    # Fetch related news and generate response
    related_news = fetch_related_news(stock_ticker)
    response = generate_rag_response(user_query, related_news)
    
    print(f"🤖 Response:\n{response}")
