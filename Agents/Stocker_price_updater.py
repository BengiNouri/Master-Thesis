import os
import yfinance as yf
from datetime import datetime
import firebase_admin
from firebase_admin import firestore, credentials
from dotenv import load_dotenv

load_dotenv()

def initialize_firebase():
    """
    Initialize Firebase and return a Firestore client.
    """
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    import firebase_admin
    from firebase_admin import credentials, firestore
    if not firebase_admin._apps:
        cred_path = primary_path if os.path.exists(primary_path) else fallback_path
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"❌ Firebase credentials not found in {cred_path}")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = initialize_firebase()

# Define stock mapping; you can import this from your other module if preferred
STOCK_MAPPING = {
    "tesla": "TSLA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "novo nordisk": "NVO",
}

def update_stock_prices():
    """
    For each stock ticker in STOCK_MAPPING, fetch the latest and previous closing prices,
    and update Firestore in the 'stock_prices' collection.
    """
    for stock_name, stock_ticker in STOCK_MAPPING.items():
        try:
            ticker = yf.Ticker(stock_ticker)
            hist = ticker.history(period="2d")
            if hist.empty or len(hist) < 2:
                print(f"⚠️ Not enough historical data for {stock_ticker}")
                continue

            latest_close = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2]

            db.collection("stock_prices").document(stock_ticker).set({
                "latest_close": latest_close,
                "previous_close": previous_close,
                "timestamp": datetime.now().isoformat()
            })
            print(f"✅ Updated stock prices for {stock_ticker}: Latest={latest_close}, Previous={previous_close}")
        except Exception as e:
            print(f"❌ Error updating stock prices for {stock_ticker}: {e}")

if __name__ == "__main__":
    update_stock_prices()
