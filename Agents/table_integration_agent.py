import firebase_admin
from firebase_admin import credentials, firestore
import os

def initialize_firebase():
    """
    Initialize Firebase with a fallback if the primary path fails.
    """
    # Path for the virtual machine
    vm_path = r"C:\MasterThesis\Keys.json"
    
    # Local machine paths
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        # Try the VM path first
        if os.path.exists(vm_path):
            cred = credentials.Certificate(vm_path)
        # Try the primary local path
        elif os.path.exists(primary_path):
            cred = credentials.Certificate(primary_path)
        # Fallback local path
        elif os.path.exists(fallback_path):
            cred = credentials.Certificate(fallback_path)
        else:
            raise FileNotFoundError("Firebase credentials file not found in any path.")
        
        firebase_admin.initialize_app(cred)

    return firestore.client()


# Initialize Firebase
db = initialize_firebase()

def link_news_to_economic_data():
    """
    Link news articles to economic data by matching keywords to stock tickers or company names.
    """
    try:
        # Fetch all economic data documents
        economic_data_docs = db.collection("latest_economic_data").stream()

        # Build a mapping of stock tickers and company names
        ticker_to_doc_id = {}
        for doc in economic_data_docs:
            data = doc.to_dict()
            stock_ticker = data.get("stock_ticker", "").upper().strip()
            company_name = data.get("long_name", "").lower().strip()

            if stock_ticker:
                ticker_to_doc_id[stock_ticker] = doc.id
            if company_name:
                ticker_to_doc_id[company_name] = doc.id

        # Fetch all news documents
        news_docs = db.collection("news").stream()

        for news_doc in news_docs:
            news_data = news_doc.to_dict()
            news_id = news_doc.id
            keywords = news_data.get("keywords", [])
            linked = False  # Flag to prevent multiple links for the same news

            for keyword in keywords:
                keyword_normalized = keyword.lower().strip()

                # Check for both ticker and company name matches
                for key, econ_doc_id in ticker_to_doc_id.items():
                    if keyword_normalized in key or key in keyword_normalized:
                        # Link news to economic data
                        db.collection("news").document(news_id).update({
                            "economic_data_id": econ_doc_id
                        })
                        print(f"✅ Linked news ID {news_id} to economic data ID {econ_doc_id}")

                        # Link back from economic data to news
                        economic_doc_ref = db.collection("latest_economic_data").document(econ_doc_id)
                        economic_doc = economic_doc_ref.get().to_dict()
                        news_links = economic_doc.get("linked_news_ids", [])

                        # Prevent duplicate links
                        if news_id not in news_links:
                            news_links.append(news_id)
                            economic_doc_ref.update({"linked_news_ids": news_links})

                        linked = True
                        break  # Stop after the first match

                if linked:
                    break  # Skip remaining keywords if already linked

            if not linked:
                print(f"⚠️ No match found for news ID {news_id}")

    except Exception as e:
        print(f"❌ Error in linking news to economic data: {e}")

if __name__ == "__main__":
    link_news_to_economic_data()
