import yfinance as yf
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

def fetch_latest_yahoo_data(stock_ticker):
    """
    Fetch the latest financial data for a given stock ticker from Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(stock_ticker)
        info = ticker.info

        if not info:
            print(f"No data found for {stock_ticker}.")
            return None

        return {
            "stock_ticker": stock_ticker,
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
        print(f"Error fetching data for {stock_ticker}: {e}")
        return None

def store_latest_data(stock_ticker, data):
    """
    Store the latest financial data in Firebase for the given stock ticker.
    """
    try:
        doc_ref = db.collection("latest_economic_data").document(stock_ticker)
        doc_ref.set(data)
        print(f"Stored data for {stock_ticker}.")
    except Exception as e:
        print(f"Error storing data: {e}")

def economic_data_agent(stock_tickers):
    """
    Fetch and store financial data for a list of stock tickers.
    """
    for ticker in stock_tickers:
        data = fetch_latest_yahoo_data(ticker)
        if data:
            store_latest_data(ticker, data)

if __name__ == "__main__":
    stock_tickers = ["TSLA", "NVO", "NVDA", "AAPL", "MSFT", "NOVO-B.CO"]
    economic_data_agent(stock_tickers)
