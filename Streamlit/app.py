import sys
import os
from datetime import datetime, timedelta
import re

import streamlit as st
import warnings
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import pandas as pd

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

# Suppress Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add root directory for custom module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# ✅ Import custom modules
from Agents.news_agent import process_articles
from Agents.rag_agent import generate_rag_response
from Agents.sentiment_agent import analyze_sentiment_and_store
from Agents.economic_data_agent import economic_data_agent
from Firebase.firestore_operations import initialize_firestore, query_news_articles, query_collection

# Initialize Firestore client
db = initialize_firestore()

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Financial Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("💼 Financial Insights Dashboard")
st.caption("Empowering your investment decisions with ML-driven insights and recommendations.")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Select Stock & Filters
# ─────────────────────────────────────────────────────────────────────────────
supported_stocks = {
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "NVO":  "Novo Nordisk",
    "AAPL":"Apple",
    "MSFT":"Microsoft"
}

st.sidebar.header("🔍 Select Stock & Date Range")
ticker_input = st.sidebar.selectbox(
    "Choose a stock:",
    options=list(supported_stocks.keys()),
    format_func=lambda s: f"{supported_stocks[s]} ({s})",
    index=1  # default to Tesla
)
start_date   = st.sidebar.date_input("📅 Start Date", datetime.today() - timedelta(days=30))
end_date     = st.sidebar.date_input("📅 End Date", datetime.today())
max_articles = st.sidebar.number_input("📄 Max Articles to Fetch", min_value=1, max_value=50, value=10)
st.sidebar.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Run Full Analysis Pipeline
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.subheader("🚀 Run Full Pipeline")
if st.sidebar.button("Start Full Pipeline"):
    with st.spinner("Running the full analysis workflow..."):
        try:
            economic_data_agent([ticker_input])
            st.success("✅ Economic data fetched successfully.")
            process_articles([ticker_input], max_articles)
            st.success("✅ News articles fetched & stored.")
            analyze_sentiment_and_store()
            st.success("✅ Sentiment analysis completed & linked to news articles.")
            st.balloons()
            st.success("🎉 Full analysis pipeline completed successfully!")
        except Exception as e:
            st.error("❌ Error during the full analysis workflow.")
            st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# Visualization: Sentiment Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_sentiment_distribution():
    try:
        snaps = (
            db.collection("sentiment_analysis")
              .order_by("timestamp", direction="DESCENDING")
              .limit(100)
              .stream()
        )
        data = [d.to_dict() for d in snaps]
        if not data:
            st.warning("⚠️ No sentiment data found.")
            return
        df = pd.DataFrame(data)
        counts = df['label'].value_counts()
        st.subheader("📊 Sentiment Distribution (last 100)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)
    except ResourceExhausted:
        st.error("❌ Firestore quota exceeded when loading sentiment data.")

# ─────────────────────────────────────────────────────────────────────────────
# Visualization: Recommendation Accuracy
# ─────────────────────────────────────────────────────────────────────────────
def plot_recommendation_accuracy():
    try:
        snaps = (
            db.collection("model_recommendations")
              .order_by("timestamp", direction="DESCENDING")
              .limit(100)
              .stream()
        )
        data = [d.to_dict() for d in snaps]
        if not data:
            st.warning("⚠️ No recommendation data found.")
            return
        df = pd.DataFrame(data)
        counts = df['is_correct'].value_counts()
        st.subheader("✅ Recommendation Accuracy (last 100)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Correct", "Incorrect"], counts.reindex([True, False], fill_value=0))
        ax.set_ylabel("Count")
        st.pyplot(fig)
    except ResourceExhausted:
        st.error("❌ Firestore quota exceeded when loading recommendation accuracy.")

st.sidebar.subheader("📈 Visualization Options")
if st.sidebar.button("Show Sentiment Distribution"):
    plot_sentiment_distribution()
if st.sidebar.button("Show Recommendation Accuracy"):
    plot_recommendation_accuracy()

# ─────────────────────────────────────────────────────────────────────────────
# Chatbot: GPT-Powered Financial Insights with Ticker Override
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🤖 Financial Insights Chatbot")

user_input = st.text_input(
    "💬 Ask our GPT-powered bot about the market",
    key="chat_query",
    placeholder="Type your question and press Enter..."
)

if user_input:
    st.write(f"**Your Query:** {user_input}")

    # 1) Start with the sidebar selection
    effective_ticker = ticker_input

    # 2) If the question mentions one of our five stocks by name or symbol, override:
    for sym, name in supported_stocks.items():
        if sym in user_input.upper() or name.lower() in user_input.lower():
            effective_ticker = sym
            break

    st.write(f"… fetching recent news for {supported_stocks[effective_ticker]} ({effective_ticker})")

    with st.spinner("Thinking…"):
        try:
            # Build ISO timestamps
            start_ts = start_date.isoformat() + "T00:00:00Z"
            end_ts   = end_date.isoformat()   + "T23:59:59Z"

            # Primary, indexed server-side fetch
            articles = query_news_articles(
                ticker=effective_ticker,
                start=start_ts,
                end=end_ts,
                limit=max_articles
            )

            # Fallback: single-field fetch + in-memory date filter
            if not articles:
                snaps = query_collection(
                    "news",
                    where_clauses=[("economic_data_id", "==", effective_ticker)],
                    limit=max_articles * 2
                )
                raw = [s.to_dict() for s in snaps]
                articles = [
                    d for d in raw
                    if start_ts <= d.get("publishedAt", "") <= end_ts
                ][:max_articles]

            st.write(f"→ retrieved {len(articles)} articles for {effective_ticker}")

            if not articles:
                final_response = f"⚠️ No news for {supported_stocks[effective_ticker]} in that date range."
            else:
                # Convert to dicts and pass to your RAG agent
                docs = [a if isinstance(a, dict) else a.__dict__ for a in articles]
                agg, rec, reason, _ = generate_rag_response(user_input, docs)
                final_response = (
                    f"**Aggregator Recommendation:** {agg}\n\n"
                    f"**GPT Recommendation:** {rec}\n\n"
                    f"**Reasoning:** {reason}"
                )

        except ResourceExhausted:
            final_response = "❌ Firestore quota exceeded when fetching news."
        except Exception as e:
            final_response = f"❌ Error generating response: {e}"

    st.write(final_response)
