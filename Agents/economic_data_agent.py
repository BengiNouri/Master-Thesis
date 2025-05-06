import os
import time
import logging
from datetime import datetime, timezone, timedelta

import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings (if present)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ─────────────────────────────────────────────────────────────────────────────
# 🔐 Initialize Firebase
# ─────────────────────────────────────────────────────────────────────────────
def initialize_firebase():
    """
    Initialize Firebase using fallback credential paths.
    Returns a Firestore client.
    """
    paths = [
        r"C:\MasterThesis\Keys.json",
        r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
        r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
    ]
    if not firebase_admin._apps:
        for p in paths:
            if os.path.exists(p):
                cred = credentials.Certificate(p)
                firebase_admin.initialize_app(cred)
                break
        else:
            raise FileNotFoundError("No Firebase credentials found.")
    return firestore.client()

# Initialize Firestore
db = initialize_firebase()

# ─────────────────────────────────────────────────────────────────────────────
# ⏱ Caching Logic
# ─────────────────────────────────────────────────────────────────────────────
CACHE_TTL = timedelta(hours=6)

def is_cache_valid(doc_ref) -> bool:
    """
    Check if existing Firestore doc has 'fetched_at' within TTL.
    """
    try:
        snap = doc_ref.get()
        if snap.exists:
            data = snap.to_dict()
            fetched = data.get("fetched_at")
            if fetched:
                fetched_dt = datetime.fromisoformat(fetched)
                if datetime.now(timezone.utc) - fetched_dt < CACHE_TTL:
                    return True
        return False
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 Retry Decorator for yfinance
# ─────────────────────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
def fetch_yahoo_info(stock_ticker: str) -> dict:
    """
    Fetch ticker.info from yfinance with retry/backoff.
    """
    ticker = yf.Ticker(stock_ticker)
    return ticker.info

# ─────────────────────────────────────────────────────────────────────────────
# 📈 Fetch & Store Economic Data
# ─────────────────────────────────────────────────────────────────────────────
def fetch_latest_yahoo_data(stock_ticker: str) -> dict:
    """
    Retrieve and validate essential financial fields, or fallback on error.
    """
    doc_ref = db.collection("latest_economic_data").document(stock_ticker.upper())
    # Caching: skip if recent
    if is_cache_valid(doc_ref):
        logger.info(f"Cache valid for {stock_ticker}, skipping API call.")
        return None

    try:
        info = fetch_yahoo_info(stock_ticker)
        # Required fields
        required = ["currentPrice", "marketCap"]
        if any(info.get(f) is None for f in required):
            logger.warning(f"Missing critical data for {stock_ticker}: {required}")
            raise ValueError("Critical data missing")

        data = {
            "stock_ticker":    stock_ticker.upper(),
            "long_name":       info.get("longName"),
            "sector":          info.get("sector"),
            "industry":        info.get("industry"),
            "current_price":   info.get("currentPrice"),
            "previous_close":  info.get("previousClose"),
            "market_cap":      info.get("marketCap"),
            "volume":          info.get("volume"),
            "52_week_high":    info.get("fiftyTwoWeekHigh"),
            "52_week_low":     info.get("fiftyTwoWeekLow"),
            "dividend_yield":  info.get("dividendYield"),
            "beta":            info.get("beta"),
            # Track fetch time
            "fetched_at":      datetime.now(timezone.utc).isoformat(),
            "status":          "ok"
        }
        return data

    except Exception as e:
        logger.error(f"Error fetching data for {stock_ticker}: {e}")
        # Placeholder doc
        return {
            "stock_ticker": stock_ticker.upper(),
            "status":       "error",
            "error_msg":    str(e),
            "fetched_at":   datetime.now(timezone.utc).isoformat()
        }


def store_latest_data(stock_ticker: str, data: dict):
    """
    Store or update the economic data document in Firestore.
    """
    try:
        db.collection("latest_economic_data").document(stock_ticker.upper()).set(data)
        logger.info(f"Stored data for {stock_ticker.upper()}")
    except Exception as e:
        logger.error(f"Error storing data for {stock_ticker.upper()}: {e}")


def economic_data_agent(stock_tickers: list[str]):
    """
    Fetch and store data for multiple tickers, with caching and error handling.
    """
    for ticker in stock_tickers:
        data = fetch_latest_yahoo_data(ticker)
        if data:
            store_latest_data(ticker, data)
        else:
            logger.info(f"Skipped storing for {ticker} (cache hit or no update)")


def refresh_economic_data(stock_tickers: list[str]):
    """
    Alias to force-refresh all data (ignores cache).
    """
    # Invalidate cache by deleting 'fetched_at'
    for ticker in stock_tickers:
        doc_ref = db.collection("latest_economic_data").document(ticker.upper())
        doc_ref.update({"fetched_at": firestore.DELETE_FIELD})
    economic_data_agent(stock_tickers)

# ─────────────────────────────────────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tickers = ["TSLA", "AAPL", "MSFT", "NVDA", "NVO"]
    economic_data_agent(tickers)
