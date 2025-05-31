import os
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials
from datetime import datetime, timezone
# ─────────────────────────────────────────────────────────────────────────────
# External data sources
import yfinance as yf
from tiingo import TiingoClient

# Agents
from Agents.news_agent import process_articles
from Agents.rag_agent import generate_rag_response
from Agents.sentiment_agent import analyze_sentiment_and_store, migrate_sentiment

import warnings
import tensorflow as tf
import torch
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import os, warnings, logging
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# Configurations
# ─────────────────────────────────────────────────────────────────────────────
VERBOSE = True
EXPERIMENT_START_DATE = datetime(2025, 1, 22, tzinfo=timezone.utc)  # ← add tzinfo
RETRY_SLEEP_SECONDS = 60
TIINGO_BACKFILL_DAYS = 7

# ─────────────────────────────────────────────────────────────────────────────
# Load environment & init clients
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("❌ OPENAI_API_KEY not set.")

# Tiingo
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise RuntimeError("❌ TIINGO_API_KEY not set in environment.")
_tiingo_client = TiingoClient({"api_key": TIINGO_API_KEY, "session": True})

# ─────────────────────────────────────────────────────────────────────────────
# Suppress TF/torch warnings
# ─────────────────────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
tf.get_logger().setLevel(logging.ERROR)       # hide Python-side warnings

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Firebase
# ─────────────────────────────────────────────────────────────────────────────
def initialize_firebase():
    paths = [
        r"C:\MasterThesis\Keys.json",
        r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
        r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
    ]
    if not firebase_admin._apps:
        for p in paths:
            if os.path.exists(p):
                cred = credentials.Certificate(p)
                break
        else:
            raise FileNotFoundError("Firebase credentials not found.")
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = initialize_firebase()

# ─────────────────────────────────────────────────────────────────────────────
# Price fetching: Tiingo first, then yfinance
# ─────────────────────────────────────────────────────────────────────────────
class PriceFetchError(Exception): pass

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def _fetch_with_yf(ticker):
    data = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
    if len(data) < 2:
        raise PriceFetchError(f"Insufficient yfinance data for {ticker}")
    return float(data.iloc[-1]), float(data.iloc[-2])


def _fetch_with_tiingo(ticker):
    end = datetime.utcnow().date()
    start = end - timedelta(days=TIINGO_BACKFILL_DAYS)
    prices = _tiingo_client.get_ticker_price(
        ticker,
        startDate=start.isoformat(),
        endDate=end.isoformat(),
        frequency='daily'
    )
    if not prices or len(prices) < 2:
        raise PriceFetchError(f"Insufficient Tiingo data for {ticker}")
    latest = prices[-1]['adjClose']
    prev = prices[-2]['adjClose']
    return float(latest), float(prev)


def fetch_closing_prices(ticker):
    """
    Primary: Tiingo; fallback: yfinance
    """
    try:
        return _fetch_with_tiingo(ticker)
    except Exception as ti:
        if VERBOSE:
            print(f"⚠️ Tiingo failed for {ticker}: {ti}. Falling back to yfinance...")
    try:
        return _fetch_with_yf(ticker)
    except Exception as yf:
        raise PriceFetchError(f"Both price sources failed for {ticker}: ti={ti}, yf={yf}")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: link
# ─────────────────────────────────────────────────────────────────────────────
def link_news_to_economic_data(news_id, stock):
    db.collection("latest_economic_data").document(stock).update({
        "linked_news_ids": firestore.ArrayUnion([news_id])
    })
    db.collection("news").document(news_id).update({"economic_data_id": stock})

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate recommendation correctness
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(stock, rec, model_name, latest, prev):
    moved   = latest > prev
    # only buy / sell are valid now
    correct = (rec.lower() == "buy"  and moved) or \
              (rec.lower() == "sell" and not moved)
    arrow   = "↑" if moved else "↓"
    pct     = (latest - prev) / prev * 100 if prev else 0
    print(f"{stock} [{model_name}]: {prev:.2f}→{latest:.2f} "
          f"({arrow}{pct:.2f}%) rec={rec} → {correct}")
    return correct

from datetime import datetime, timezone   # make sure this is at the top …

# ─────────────────────────────────────────────────────────────────────────────
# Store recommendation + economic data (UTC-aware)
# ─────────────────────────────────────────────────────────────────────────────
def store_recommendation(
    stock: str,
    agg_rec: str,
    gpt_rec: str,
    summary: dict,
    corr: bool,
    latest: float,
    prev: float,
    detail: str
) -> None:
    # 1️⃣  figure out “experiment day” in a tz-safe way
    now_utc = datetime.now(timezone.utc)
    start   = EXPERIMENT_START_DATE
    if start.tzinfo is None:               # legacy naïve constant
        start = start.replace(tzinfo=timezone.utc)
    experiment_day = (now_utc - start).days

    # 2️⃣  shared timestamp string
    ts_str = now_utc.isoformat()

    # 3️⃣  build and store recommendation doc
    rec_doc = {
        "stock_ticker":            stock,
        "sentiment_summary":       summary,
        "latest_close":            latest,
        "previous_close":          prev,
        "is_correct":              corr,
        "aggregator_recommendation": agg_rec,
        "gpt_recommendation":      gpt_rec,
        "timestamp":               ts_str,
        "experiment_day":          experiment_day,
        "recommendation_detail":   detail
    }
    rec_ref = db.collection("model_recommendations").add(rec_doc)
    rec_id  = rec_ref[1].id
    print(f"✅ Stored rec {rec_id} for {stock}")

    # 4️⃣  build and store economic-data doc
    econ_doc = {
        "stock_ticker":   stock,
        "experiment_day": experiment_day,
        "latest_close":   latest,
        "previous_close": prev,
        "price_change":   (latest - prev) / prev * 100 if prev else None,
        "sentiment_summary": summary,
        "recommendation_id": rec_id,
        "timestamp":      ts_str
    }
    db.collection("economic_data").add(econ_doc)
    print(f"✅ Stored econ for {stock}")


# ─────────────────────────────────────────────────────────────────────────────
# Daily pipeline – FinBERT vs GPT-4o-mini vs actual price move
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime, timedelta, timezone
from rich.console import Console
from rich.table   import Table

WINDOW_HOURS        = 24         # rolling horizon (7 days)
MAX_DOCS_PER_STOCK  = 20           # cap after sorting by ingested_at
REQUIRE_TODAY_NEWS  = True        # True → must be ingested today
SHOW_ARTICLE_LINES  = False        # verbose per-headline print

console = Console()

def finbert_to_rec(total_signed: float) -> str:
    """Aggregate FinBERT score → BUY / SELL."""
    return "buy" if total_signed > 0 else "sell"


def run_daily_pipeline(stocks, articles_per_stock: int = 20) -> None:
    cutoff    = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    today_utc = cutoff.date()
    summary_rows = []

    for stock in stocks:
        console.rule(f"[bold yellow]{stock} (last {WINDOW_HOURS} h)")

        # 1) ingest up to `articles_per_stock` new headlines
        process_articles([stock], articles_per_stock)

        # 2) ensure every news doc links to its econ record
        for snap in db.collection("news") \
                      .where("keywords", "array_contains", stock).stream():
            if not snap.to_dict().get("economic_data_id"):
                link_news_to_economic_data(snap.id, stock)

        analyze_sentiment_and_store()
        migrate_sentiment()

        # 3) collect docs inside window – then cap to 20 freshest
        docs = []
        for snap in db.collection("news") \
                      .where("economic_data_id", "==", stock).stream():
            d = snap.to_dict()
            if "sentiment_score" not in d:
                continue

            raw_ts = d.get("ingested_at") or d.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except Exception:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            fresh    = ts >= cutoff
            is_today = ts.date() == today_utc
            if fresh and (not REQUIRE_TODAY_NEWS or is_today):
                d["__ts"] = ts            # stash for sorting
                docs.append(d)

        # keep only N freshest by ingested_at
        docs.sort(key=lambda x: x["__ts"], reverse=True)
        docs = docs[:MAX_DOCS_PER_STOCK]

        if not docs:
            console.print(
                f"[italic]No qualifying headlines "
                f"(≤{WINDOW_HOURS} h and REQUIRE_TODAY_NEWS={REQUIRE_TODAY_NEWS}).[/]")
            continue

        # 4) (optional) per-headline sentiment
        signed_scores = []
        if SHOW_ARTICLE_LINES:
            console.print("• Article sentiment:")
        for d in docs:
            lbl  = d["sentiment_label"].lower()
            sc   = d["sentiment_score"]
            val  = +sc if lbl == "positive" else -sc if lbl == "negative" else 0.0
            signed_scores.append(val)
            if SHOW_ARTICLE_LINES:
                console.print(f"  {val:+.3f} ({lbl})  {d['title'][:80]}")

        # 5) model outputs
        finbert_total = sum(signed_scores)
        finbert_rec   = finbert_to_rec(finbert_total)
        agg_rec, gpt_rec, detail, summary = generate_rag_response(
            f"Outlook for {stock}?", docs
        )

        # 6) prices
        try:
            latest, prev = fetch_closing_prices(stock)
        except PriceFetchError as err:
            console.print(f"[red]{err}[/]")
            continue
        move_pct = (latest - prev) / prev * 100
        arrow    = "↑" if latest > prev else "↓"

        # 7) accuracy flags
        fin_ok = evaluate_model(stock, finbert_rec, "FinBERT",    latest, prev)
        gpt_ok = evaluate_model(stock, gpt_rec,    "GPT-4o-mini", latest, prev)

        # 8) add to summary table
        summary_rows.append((
            stock,
            f"{finbert_total:+.3f}",
            finbert_rec.upper(),
            gpt_rec.upper(),
            f"{prev:.2f}",
            f"{latest:.2f}",
            f"{move_pct:+.2f} %",
            "✅" if fin_ok else "❌",
            "✅" if gpt_ok else "❌"
        ))

        # 9) write to Firestore
        store_recommendation(
            stock, agg_rec, gpt_rec, summary, gpt_ok,
            latest, prev, detail
        )

    # ── pretty Rich table ────────────────────────────────────────────
    if summary_rows:
        table = Table(title="Daily Model Summary", show_lines=True)
        for col in ["Ticker", "FinBERT Σ", "FinBERT Rec", "GPT Rec",
                    "Prev Close", "Latest Close", "Move %", "FinBERT✔", "GPT✔"]:
            table.add_column(col, justify="right")
        for row in summary_rows:
            table.add_row(*row)
        console.print(table)

    console.print("[bold green]✅ Daily pipeline complete.[/]")


if __name__ == "__main__":
    run_daily_pipeline(["TSLA", "AAPL", "MSFT", "NVDA", "NVO"])
