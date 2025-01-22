import sys
import os
from datetime import datetime, timedelta

import streamlit as st
import warnings
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import pandas as pd

from dotenv import load_dotenv

# Suppress Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# ✅ Import custom modules
from Agents.news_agent import process_articles
from Agents.rag_agent import generate_rag_response
from Agents.sentiment_agent import analyze_sentiment_and_store
from Agents.economic_data_agent import economic_data_agent
from Firebase.firestore_operations import query_firestore, initialize_firestore

# Explicitly initialize Firestore client
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
# Sidebar - Search / Filter Options
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("🔍 Search & Analysis Options")
query = st.sidebar.text_input("Search Keyword or Stock Ticker", "Tesla")
start_date = st.sidebar.date_input("📅 Start Date", datetime.today() - timedelta(days=30))
end_date = st.sidebar.date_input("📅 End Date", datetime.today())
max_articles = st.sidebar.number_input("📄 Max Articles to Fetch", min_value=1, max_value=50, value=10)
st.sidebar.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Run Full Analysis Workflow
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.subheader("🚀 Run Full Analysis")
if st.sidebar.button("Start Full Pipeline"):
    with st.spinner("Running the full analysis workflow..."):
        try:
            # 1️⃣ Fetch Economic Data
            economic_data_agent([query])
            st.success("✅ Economic data fetched successfully.")

            # 2️⃣ Fetch News Articles
            process_articles([query], max_articles)
            st.success("✅ News articles fetched & stored.")

            # 3️⃣ Run Sentiment Analysis
            analyze_sentiment_and_store()
            st.success("✅ Sentiment analysis completed & linked to news articles.")

            st.balloons()  # Small celebratory effect
            st.success("🎉 Full analysis workflow completed successfully!")
        except Exception as e:
            st.error("❌ Error during the full analysis workflow.")
            st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# Visualization - Sentiment Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_sentiment_distribution():
    try:
        sentiments = db.collection("sentiment_analysis").stream()
        sentiment_data = [doc.to_dict() for doc in sentiments]
        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            counts = df['label'].value_counts()
            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.warning("⚠️ No sentiment data found for visualization.")
    except Exception as e:
        st.error("❌ Error loading sentiment distribution.")
        st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# Visualization - Recommendation Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_recommendation_accuracy():
    try:
        recommendations = db.collection("model_recommendations").stream()
        rec_data = [doc.to_dict() for doc in recommendations]
        if rec_data:
            df = pd.DataFrame(rec_data)
            accuracy_counts = df['is_correct'].value_counts()
            labels = ["Correct", "Incorrect"]
            st.subheader("✅ Recommendation Accuracy")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, accuracy_counts, color=["green", "red"])
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning("⚠️ No recommendation data available for visualization.")
    except Exception as e:
        st.error("❌ Error loading recommendation accuracy.")
        st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Visualization Options
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.subheader("📈 Visualization Options")
if st.sidebar.button("Show Sentiment Distribution"):
    plot_sentiment_distribution()
if st.sidebar.button("Show Recommendation Accuracy"):
    plot_recommendation_accuracy()

# ─────────────────────────────────────────────────────────────────────────────
# Simplified ChatGPT-Style Conversation & Recommendation Presentation
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.header("🤖 Financial Insights Chatbot")

# Chatbot interface for user queries
user_input = st.text_input("💬 Ask our GPT-powered bot about the market", key="chat_query", placeholder="Type your question and press Enter...")

if user_input:
    st.write(f"**Your Query:** {user_input}")
    with st.spinner("Thinking..."):
        try:
            documents = query_firestore("news")
            if not documents:
                final_response = "⚠️ No news data available for generating a response."
            else:
                aggregator_rec, gpt_rec, _ = generate_rag_response(user_input, documents)
                final_response = (
                    f"**Aggregator Recommendation:** {aggregator_rec}\n\n"
                    f"**GPT Recommendation:** {gpt_rec}"
                )
        except Exception as e:
            final_response = f"❌ Error generating response: {e}"
    st.write(final_response)
