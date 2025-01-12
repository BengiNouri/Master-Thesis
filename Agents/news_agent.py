import os
import sys
import requests
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

# Firebase Initialization Function
def initialize_firebase():
    """
    Initialize Firebase if not already initialized.
    """
    if not firebase_admin._apps:
        cred = credentials.Certificate(r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json")
        initialize_app(cred)
    return firestore.client()

# Initialize Firebase
db = initialize_firebase()

# News API Configuration
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# 🔥 Dynamic Mapping between company names and stock tickers
def build_stock_mapping():
    """
    Dynamically build a mapping between company names and their stock tickers from Firestore.
    """
    stock_mapping = {}
    try:
        # Fetch all documents from 'latest_economic_data' collection
        docs = db.collection("latest_economic_data").stream()
        for doc in docs:
            data = doc.to_dict()
            company_name = data.get("long_name", "").lower()  # Example: "Apple Inc."
            stock_ticker = data.get("stock_ticker", "").upper()  # Example: "AAPL"

            if company_name and stock_ticker:
                stock_mapping[company_name] = stock_ticker
                stock_mapping[stock_ticker.lower()] = stock_ticker  # Include ticker as a key too

        print("✅ Stock mapping loaded from Firestore.")
    except Exception as e:
        print(f"❌ Error building stock mapping: {e}")
    
    return stock_mapping

# Build the dynamic mapping
STOCK_MAPPING = build_stock_mapping()

def fetch_news_articles(keywords, page_size=10):
    """
    Fetch news articles from NewsAPI.
    """
    try:
        if not NEWS_API_KEY:
            raise ValueError("NEWS_API_KEY is not set. Check your .env file or environment variables.")

        query = " OR ".join(keywords)
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
        }
        response = requests.get(NEWS_API_ENDPOINT, params=params)
        response.raise_for_status()

        articles = response.json().get("articles", [])
        return articles
    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return []

def store_in_firebase(collection_name, data):
    """
    Store documents in Firebase Firestore.
    """
    try:
        for item in data:
            doc_ref = db.collection(collection_name).document()
            item["doc_id"] = doc_ref.id  # Save the document ID for linking
            doc_ref.set(item)
        print(f"✅ Stored {len(data)} documents in the '{collection_name}' collection.")
    except Exception as e:
        print(f"❌ Error storing data in Firestore: {e}")

def link_news_to_economic_data(news_id, keyword):
    """
    Dynamically link news articles to economic data based on the keyword.
    """
    try:
        # Convert the keyword to lowercase for a case-insensitive search
        stock_ticker = STOCK_MAPPING.get(keyword.lower(), None)
        
        if not stock_ticker:
            print(f"⚠️ No stock ticker found for keyword '{keyword}'. Skipping linking.")
            return

        # Link the news article to the correct economic data
        economic_ref = db.collection("latest_economic_data").document(stock_ticker)
        economic_ref.update({
            "linked_news_ids": firestore.ArrayUnion([news_id])
        })

        # Update the news document with the economic data ID
        news_ref = db.collection("news").document(news_id)
        news_ref.update({"economic_data_id": stock_ticker})

        print(f"🔗 Linked news ID {news_id} to economic data ID {stock_ticker}")

    except Exception as e:
        print(f"❌ Error linking news to economic data: {e}")

def process_articles(keywords, page_size=10):
    """
    Fetch, process, and store articles, then trigger sentiment analysis and linking.
    """
    print(f"🔎 Fetching news articles for: {keywords}")
    articles = fetch_news_articles(keywords, page_size)

    if not articles:
        print(f"⚠️ No articles found for keywords: {keywords}.")
        return

    # Step 1: Structure and Store Articles
    structured_articles = [
        {
            "title": article.get("title"),
            "content": article.get("content"),
            "url": article.get("url"),
            "publishedAt": article.get("publishedAt"),
            "source": article.get("source", {}).get("name"),
            "keywords": keywords,
            "sentiment_id": None,
            "economic_data_id": None,
        }
        for article in articles
    ]

    store_in_firebase("news", structured_articles)

    # Step 2: Trigger Sentiment Analysis
    print("🧠 Triggering sentiment analysis...")
    analyze_sentiment_and_store()

    # Step 3: Dynamically Link Articles to Economic Data
    print("🔗 Linking news articles to economic data...")
    for article in structured_articles:
        for keyword in keywords:
            link_news_to_economic_data(article["doc_id"], keyword)

    print("✅ Workflow completed successfully!")

if __name__ == "__main__":
    # Dynamic user input
    keywords = ["Tesla?", "TSLA"]  # Example test with Apple
    page_size = 5

    # Run the data pipeline
    process_articles(keywords, page_size)
