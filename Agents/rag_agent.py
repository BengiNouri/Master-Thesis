import os
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf
import firebase_admin
from firebase_admin import firestore, credentials
from openai import OpenAI
from httpx import ReadTimeout
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─────────────────────────────────────────────────────────────────────────────
# Load environment variables & initialize OpenAI client
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in environment variables")

client = OpenAI(api_key=api_key)

# ─────────────────────────────────────────────────────────────────────────────
# Firebase initialization
# ─────────────────────────────────────────────────────────────────────────────
def initialize_firebase():
    """
    Initialize Firebase and return a Firestore client.
    """
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    if not firebase_admin._apps:
        cred_path = primary_path if os.path.exists(primary_path) else fallback_path
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"❌ Firebase credentials not found in {cred_path}")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

# Singleton Firestore client
db = initialize_firebase()

# ─────────────────────────────────────────────────────────────────────────────
# Utility: resolve ticker from name or symbol
# ─────────────────────────────────────────────────────────────────────────────
def resolve_stock_ticker(name_or_ticker: str) -> str:
    key = name_or_ticker.strip().upper()
    doc = db.collection("latest_economic_data").document(key).get()
    if doc.exists:
        return key
    snaps = (
        db.collection("latest_economic_data")
          .where("long_name", "==", name_or_ticker.lower().strip())
          .limit(1)
          .get()
    )
    if snaps:
        return snaps[0].id
    return key

# ─────────────────────────────────────────────────────────────────────────────
# Fetch closing prices with retry on rate-limits
# ─────────────────────────────────────────────────────────────────────────────
class RateLimitError(Exception):
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
def fetch_closing_prices(stock_ticker: str):
    """
    Return (latest_close, previous_close) or (None, None) if unavailable.
    Retries up to 3 times on 429 errors.
    """
    try:
        ticker = yf.Ticker(stock_ticker)
        hist = ticker.history(period="5d")
        if hist.empty or len(hist) < 2:
            print(f"⚠️ Not enough historical data for {stock_ticker}")
            return None, None
        return hist['Close'].iloc[-1], hist['Close'].iloc[-2]
    except Exception as e:
        msg = str(e)
        if "429" in msg or "Rate limited" in msg:
            raise RateLimitError(f"Rate-limited on {stock_ticker}")
        logging.error(f"❌ Error fetching closing prices for {stock_ticker}: {e}")
        return None, None

# ─────────────────────────────────────────────────────────────────────────────
# Fetch related news from Firestore
# ─────────────────────────────────────────────────────────────────────────────
def fetch_related_news(stock_ticker: str):
    """
    Pull all news articles linked to the given economic_data document.
    """
    econ_doc = db.collection("latest_economic_data").document(stock_ticker).get()
    if not econ_doc.exists:
        print(f"⚠️ No economic data for {stock_ticker}")
        return []
    linked_ids = econ_doc.to_dict().get("linked_news_ids", [])
    articles = []
    for nid in linked_ids:
        ndoc = db.collection("news").document(nid).get()
        if ndoc.exists:
            articles.append(ndoc.to_dict())
    return articles

# ─────────────────────────────────────────────────────────────────────────────
# Main RAG: recommendation + reasoning
# ─────────────────────────────────────────────────────────────────────────────
def generate_rag_response(query: str, documents: list):
    """
    Returns (aggregator_rec, gpt_rec, reasoning, sentiment_summary).
    Reasoning is exactly two bullets with bolded score and label.
    """
    try:
        # 1️⃣ Aggregate sentiment scores
        summary = {"positive":0.0, "neutral":0.0, "negative":0.0}
        for d in documents:
            lbl = d.get("sentiment_label", "neutral").lower()
            score = float(d.get("sentiment_score", 0.0))
            summary[lbl] += score

        # 2️⃣ Aggregator logic
        if summary["positive"] > summary["negative"]:
            agg = "Buy"
        elif summary["negative"] > summary["positive"]:
            agg = "Sell"
        else:
            agg = "Hold"

        # 3️⃣ Build context snippet (up to 4)
        ctx_lines = []
        for d in documents[:4]:
            title = d.get("title", "No Title")
            score = float(d.get("sentiment_score",0.0))
            label = d.get("sentiment_label", "Neutral")
            ctx_lines.append(f"- {title} (**{score:.4f}**, {label})")
        ctx = "\n".join(ctx_lines)

        # 4️⃣ Strict prompt format
        prompt = (
            "You are a seasoned financial analyst.\n\n"
            f"Aggregator signal: {agg}\n\n"
            "Top articles (title, bold score, label):\n"
            f"{ctx}\n\n"
            "Answer in exactly this format:\n"
            "Recommendation: <Buy/Sell/Hold>  (one sentence)\n"
            "Reasoning:\n"
            "- <Title 1> (**X.XXXX**, Label): …how this supports your view\n"
            "- <Title 2> (**Y.YYYY**, Label): …how this supports your view"
        )

        logging.info("🛰️ Sending prompt to OpenAI…")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0,
            timeout=15
        )
        out = resp.choices[0].message.content.strip()
        logging.info(f"🛰️ OpenAI returned: {out!r}")

        # 5️⃣ Parse recommendation
        rec_line = next((l for l in out.splitlines()
                         if l.lower().startswith("recommendation:")), "")
        rec = rec_line.split(":",1)[-1].strip().capitalize() or agg

        # 6️⃣ Parse two reasoning bullets
        bullets = [l for l in out.splitlines() if l.strip().startswith("- ")][:2]
        reasoning = "\n".join(bullets) if bullets else "No reasoning."

        return agg, rec, reasoning, summary

    except ReadTimeout:
        logging.error("⏰ OpenAI request timed out")
        return agg, "Hold", "- **AI timed out**—please try again.", summary
    except Exception as e:
        logging.exception("❌ RAG agent failed")
        return "Hold", "Hold", "- **Error generating reasoning**.", {"positive":0, "neutral":0, "negative":0}

# ─────────────────────────────────────────────────────────────────────────────
# Store recommendation in Firestore
# ─────────────────────────────────────────────────────────────────────────────
def store_recommendation(stock_ticker: str, aggregator_rec: str, gpt_rec: str, sentiment_sum: dict):
    try:
        db.collection("recommendations").document(stock_ticker).set({
            "aggregator_rec": aggregator_rec,
            "gpt_rec": gpt_rec,
            "sentiment_sum": sentiment_sum,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"✅ Stored recommendation for {stock_ticker}: {aggregator_rec} | {gpt_rec}")
    except Exception as e:
        logging.error(f"❌ Error storing recommendation for {stock_ticker}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Example / debug run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user_query   = "What is the latest news about Tesla?"
    stock_ticker = resolve_stock_ticker("Tesla")
    related_news = fetch_related_news(stock_ticker)

    agg, gpt, reason, summary = generate_rag_response(user_query, related_news)
    print(f"Aggregator Recommendation: {agg}")
    print(f"GPT Recommendation:        {gpt}")
    print(f"Reasoning:                 {reason}")
    print(f"Sentiment Summary:         {summary}")

    latest, prev = fetch_closing_prices(stock_ticker)
    if latest and prev:
        correct = ((gpt.lower() in ["buy","hold"] and latest>prev) or
                   (gpt.lower()=="sell" and latest<prev))
        print(f"📊 Recommendation correctness: {correct}")
        print(f"Today's close: {latest}, Yesterday's: {prev}")
    else:
        print(f"⚠️ No stock price data available for {stock_ticker}.")

    store_recommendation(stock_ticker, agg, gpt, summary)
