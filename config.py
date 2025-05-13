import os
import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    """
    Initialize Firebase with fallback paths.
    """
    vm_path = r"C:\MasterThesis\Keys.json"
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    if not firebase_admin._apps:
        if os.path.exists(vm_path):
            cred = credentials.Certificate(vm_path)
        elif os.path.exists(primary_path):
            cred = credentials.Certificate(primary_path)
        elif os.path.exists(fallback_path):
            cred = credentials.Certificate(fallback_path)
        else:
            raise FileNotFoundError("Firebase credentials file not found.")
        
        firebase_admin.initialize_app(cred)

    return firestore.client()

# Initialize Firestore
db = initialize_firebase()

# 🔄 Load Stock Mapping
def load_stock_mapping():
    """
    Load stock mappings from Firestore for company names and stock tickers.
    """
    mapping = {}
    try:
        docs = db.collection("latest_economic_data").stream()
        for doc in docs:
            data = doc.to_dict()
            company_name = data.get("long_name", "").lower().strip()
            stock_ticker = doc.id.upper().strip()

            if company_name and stock_ticker:
                mapping[company_name] = stock_ticker
                mapping[stock_ticker.lower()] = stock_ticker

        print("✅ Stock mapping loaded.")
    except Exception as e:
        print(f"❌ Error loading stock mapping: {e}")
    return mapping

# 🔍 Global Stock Mapping
STOCK_MAPPING = load_stock_mapping()
