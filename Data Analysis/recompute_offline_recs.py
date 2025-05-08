# recompute_offline_recs_v3.py

import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ───────────────────────────────────────────────────────────────────────────────
# 1️⃣ PYTHONPATH setup
# ───────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# ───────────────────────────────────────────────────────────────────────────────
# 2️⃣ Load CSVs
# ───────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")

# find your hole‐free price file
candidates = [
    "economic_data_full2.csv",
    "economic_data_full.csv",
    "economic_data_full_tiingo.csv",
]
for fn in candidates:
    path = os.path.join(DATA_DIR, fn)
    if os.path.exists(path):
        price_file = path
        break
else:
    raise FileNotFoundError(f"Could not find any of {candidates!r} in {DATA_DIR!r}")

# full price history (previous_close, latest_close)
df_prices = pd.read_csv(
    price_file,
    parse_dates=["date"]
)

# news feed
df_news = pd.read_csv(
    os.path.join(DATA_DIR, "news.csv"),
    parse_dates=["publishedAt", "analyzed_at"],
)

# ───────────────────────────────────────────────────────────────────────────────
# 2b️⃣ Build per‐day price lookup
# ───────────────────────────────────────────────────────────────────────────────
for col in ("previous_close", "latest_close"):
    if col not in df_prices.columns:
        raise KeyError(f"{os.path.basename(price_file)} missing '{col}' column")

hist_lookup = (
    df_prices
    .assign(date=lambda df: df["date"].dt.date)
    .set_index(["stock_ticker", "date"])
    .sort_index()
)

# ───────────────────────────────────────────────────────────────────────────────
# 3️⃣ Import your RAG‐agent
# ───────────────────────────────────────────────────────────────────────────────
from Agents.rag_agent import generate_rag_response

# ───────────────────────────────────────────────────────────────────────────────
# 4️⃣ Main recompute loop (aggregator + GPT/RAG)
# ───────────────────────────────────────────────────────────────────────────────
STOCKS   = ["TSLA", "AAPL", "MSFT", "NVDA", "NVO"]
all_recs = []

print(f"\n▶️ Full‐history offline recompute started at {datetime.utcnow().isoformat()}Z\n")

for ticker in STOCKS:
    print(f"── {ticker} ──")
    df_t = df_news[df_news["economic_data_id"] == ticker]
    if df_t.empty:
        print(f"⚠️ no news for {ticker}")
        continue

    # only those dates for which we have price data
    available_dates = set(hist_lookup.loc[ticker].index) if ticker in hist_lookup.index.levels[0] else set()
    run_dates = [d for d in sorted(df_t["publishedAt"].dt.date.unique()) if d in available_dates]
    if not run_dates:
        print(f"  ⚠️ no matching price dates for {ticker}, skipping")
        continue

    for exp_day, run_date in enumerate(run_dates, start=1):
        day_slice = df_t[df_t["publishedAt"].dt.date == run_date]
        articles  = day_slice.to_dict("records")

        # — Aggregator (FinBERT net‐sentiment ensemble) + GPT/RAG
        agg, gpt, reasoning, summary = generate_rag_response(
            f"Outlook for {ticker} on {run_date}", articles
        )

        # skip pure‐Hold aggregator days
        if agg == "Hold":
            print(f"  {run_date} → skipping Hold")
            continue

        # if GPT returns Hold, fall back to aggregator
        if gpt == "Hold":
            gpt = agg

        # pull closes
        prev   = float(hist_lookup.at[(ticker, run_date), "previous_close"])
        latest = float(hist_lookup.at[(ticker, run_date), "latest_close"])

        # price movement direction
        if   latest > prev:   direction =  1
        elif latest < prev:   direction = -1
        else:                 direction =  0

        # correctness flags
        correct_agg = (agg == "Buy"  and direction ==  1) or \
                      (agg == "Sell" and direction == -1)
        correct_gpt = (gpt == "Buy"  and direction ==  1) or \
                      (gpt == "Sell" and direction == -1)

        all_recs.append({
            "stock_ticker":              ticker,
            "run_date":                  run_date,
            "aggregator_recommendation": agg,
            "gpt_recommendation":        gpt,
            "previous_close":            prev,
            "latest_close":              latest,
            "price_direction":           direction,
            "is_correct_agg":            correct_agg,
            "is_correct_gpt":            correct_gpt,
            "sentiment_summary":         summary,
            "timestamp":                 datetime.utcnow().isoformat() + "Z",
            "experiment_day":            exp_day,
        })

        print(
            f"  {run_date} → agg={agg} (corr={correct_agg}), "
            f"gpt={gpt} (corr={correct_gpt})"
        )

# ───────────────────────────────────────────────────────────────────────────────
# 5️⃣ Save results + print accuracies
# ───────────────────────────────────────────────────────────────────────────────
df_out  = pd.DataFrame(all_recs)
OUT_PATH = os.path.join(SCRIPT_DIR, "model_recommendations_v3.csv")
df_out.to_csv(OUT_PATH, index=False)

agg_acc = df_out["is_correct_agg"].mean()
gpt_acc = df_out["is_correct_gpt"].mean()
print(f"\n▶️ Aggregator accuracy: {agg_acc:.2%}")
print(f"▶️ GPT/RAG accuracy:      {gpt_acc:.2%}")

print(f"\n✅ Saved {len(df_out)} rows to:\n   {OUT_PATH}\n")
