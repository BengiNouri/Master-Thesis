# fetch_full_price_history_tiingo.py
import os
import pandas as pd
from datetime import timedelta
from tiingo import TiingoClient
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TIINGO_API_KEY")
if not API_KEY:
    raise RuntimeError("TIINGO_API_KEY not set in .env")

# ──────────────────────────────────────────────────────────────────────────
# 1️⃣ Read your existing economic_data.csv to get tickers & date window
# ──────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR   = os.path.join(SCRIPT_DIR, "Data")
IN_FILE    = os.path.join(DATA_DIR, "economic_data.csv")

df_hist = pd.read_csv(IN_FILE, parse_dates=["timestamp"])
df_hist["date"] = df_hist["timestamp"].dt.date
start_date = df_hist["date"].min()
end_date   = df_hist["date"].max() + timedelta(days=1)

TICKERS = sorted(df_hist["stock_ticker"].unique())
print(f"▶️  Fetching {TICKERS} from {start_date} to {end_date}")

# ──────────────────────────────────────────────────────────────────────────
# 2️⃣ Configure Tiingo client & fetch daily prices
# ──────────────────────────────────────────────────────────────────────────
config = {
    'api_key': API_KEY,
    'session': True
}
client = TiingoClient(config)

all_data = []
for ticker in TICKERS:
    print(f"  • {ticker}", end="", flush=True)
    prices = client.get_ticker_price(
        ticker,
        startDate=start_date.isoformat(),
        endDate=end_date.isoformat(),
        frequency='daily'
    )
    df = pd.DataFrame(prices)[['date', 'adjClose']].rename(
        columns={'adjClose': 'latest_close'}
    )
    df['previous_close'] = df['latest_close'].shift(1)
    df['stock_ticker']   = ticker
    all_data.append(df.dropna(subset=['previous_close']))
    print(" ✓")

full = pd.concat(all_data, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────
# 3️⃣ Write out the full, hole-free price history
# ──────────────────────────────────────────────────────────────────────────
OUT_FILE = os.path.join(DATA_DIR, "economic_data_full_tiingo.csv")
full.to_csv(OUT_FILE, index=False)
print(f"\n✅ Wrote complete price history to:\n   {OUT_FILE}\n")
