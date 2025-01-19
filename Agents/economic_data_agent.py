import os
import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ✅ Initialize Firebase
def initialize_firebase():
    """
    Initialize Firebase with a fallback if the primary path fails.
    """
    # Path for the virtual machine
    vm_path = r"C:\MasterThesis\Keys.json"
    
    # Local machine paths
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
            raise FileNotFoundError("❌ Firebase credentials file not found in any path.")

        firebase_admin.initialize_app(cred)

    return firestore.client()

# Initialize Firebase
db = initialize_firebase()

# ✅ Fetch Yahoo Finance Data
def fetch_latest_yahoo_data(stock_ticker):
    """
    Fetch the latest financial data for a given stock ticker from Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(stock_ticker)
        info = ticker.info

        # Validate essential fields
        required_fields = ["currentPrice", "previousClose", "marketCap", "sector", "industry", 
                           "volume", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "longName", "dividendYield"]

        missing_fields = [field for field in required_fields if info.get(field) is None]
        if missing_fields:
            print(f"⚠️ Missing fields for {stock_ticker}: {missing_fields}")

        # Skip storing if critical data is missing
        if not info.get("currentPrice") or not info.get("marketCap"):
            print(f"⚠️ Critical data missing for {stock_ticker}. Skipping...")
            return None

        return {
            "stock_ticker": stock_ticker.upper(),
            "long_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_price": info.get("currentPrice"),
            "previous_close": info.get("previousClose"),
            "market_cap": info.get("marketCap"),
            "volume": info.get("volume"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "news_ids": []
        }

    except Exception as e:
        print(f"❌ Error fetching data for {stock_ticker}: {e}")
        return None

# ✅ Store Data in Firestore
def store_latest_data(stock_ticker, data):
    """
    Store the latest financial data in Firebase for the given stock ticker.
    """
    try:
        db.collection("latest_economic_data").document(stock_ticker.upper()).set(data)
        print(f"✅ Stored data for {stock_ticker.upper()}")
    except Exception as e:
        print(f"❌ Error storing data for {stock_ticker.upper()}: {e}")

# ✅ Main Function to Fetch & Store Data
def economic_data_agent(stock_tickers):
    """
    Fetch and store economic data for multiple stock tickers.
    """
    for ticker in stock_tickers:
        data = fetch_latest_yahoo_data(ticker)
        if data:
            store_latest_data(ticker, data)
        else:
            print(f"⚠️ Skipping {ticker} due to missing data.")

# ✅ Refresh Data Function
def refresh_economic_data(stock_tickers):
    """
    Refresh and update existing economic data in Firestore.
    """
    print("🔄 Refreshing economic data...")
    economic_data_agent(stock_tickers)
    print("✅ Economic data refresh complete.")

# ✅ Run Script Directly
if __name__ == "__main__":
    stock_tickers = ["TSLA", "AAPL", "MSFT", "NVDA", "NVO"]
    refresh_economic_data(stock_tickers)
