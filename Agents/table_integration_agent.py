import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

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
