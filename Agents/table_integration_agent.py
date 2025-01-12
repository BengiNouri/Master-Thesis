import firebase_admin
from firebase_admin import credentials, firestore

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

def link_news_to_economic_data():
    """
    Link news articles to economic data by matching keywords to stock tickers.
    """
    try:
        # Get all economic data documents
        economic_data_docs = db.collection("latest_economic_data").stream()

        # Build a mapping of stock_ticker to document ID
        ticker_to_doc_id = {}
        for doc in economic_data_docs:
            data = doc.to_dict()
            stock_ticker = data.get("stock_ticker")
            if stock_ticker:
                ticker_to_doc_id[stock_ticker.upper()] = doc.id

        # Get all news documents
        news_docs = db.collection("news").stream()

        for news_doc in news_docs:
            news_data = news_doc.to_dict()
            news_id = news_doc.id
            keywords = news_data.get("keywords", [])

            for keyword in keywords:
                matched_ticker = ticker_to_doc_id.get(keyword.upper())
                if matched_ticker:
                    # Link news to economic data
                    db.collection("news").document(news_id).update({
                        "economic_data_id": matched_ticker
                    })
                    print(f"Linked news ID {news_id} to economic data ID {matched_ticker}")

                    # Optional: Link back from economic data to news
                    economic_doc_ref = db.collection("latest_economic_data").document(matched_ticker)
                    economic_doc = economic_doc_ref.get().to_dict()
                    news_links = economic_doc.get("linked_news_ids", [])
                    news_links.append(news_id)
                    economic_doc_ref.update({"linked_news_ids": news_links})

                    break  # Stop after the first match

    except Exception as e:
        print(f"Error in linking news to economic data: {e}")

if __name__ == "__main__":
    link_news_to_economic_data()
