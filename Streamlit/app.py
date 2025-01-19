import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit as st
import warnings
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import pandas as pd

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
from Firebase.firestore_operations import query_firestore, db

# 📊 Streamlit app configuration
st.set_page_config(page_title="📈 Financial Insights Dashboard", layout="wide")
st.title("📊 Financial Insights Dashboard")

# Sidebar - 🔍 Search Options
st.sidebar.header("🔍 Search Options")
query = st.sidebar.text_input("Search Articles or Stock Ticker", "Tesla")
start_date = st.sidebar.date_input("📅 Start Date", datetime.today() - timedelta(days=30))
end_date = st.sidebar.date_input("📅 End Date", datetime.today())
max_articles = st.sidebar.number_input("📄 Max Articles", min_value=1, max_value=50, value=10)

# 🔄 Single Button: Complete Workflow
if st.sidebar.button("🚀 Run Full Analysis"):
    with st.spinner("Running full analysis workflow..."):
        try:
            # 1️⃣ Fetch Economic Data
            economic_data_agent([query])
            st.success("✅ Economic data fetched successfully.")

            # 2️⃣ Fetch News Articles
            process_articles([query], max_articles)
            st.success("✅ News articles fetched and stored.")

            # 3️⃣ Run Sentiment Analysis
            analyze_sentiment_and_store()
            st.success("✅ Sentiment analysis completed and linked to news articles.")

            st.success("🎉 Full analysis workflow completed successfully!")

        except Exception as e:
            st.error("❌ Error during the full analysis workflow.")
            st.exception(e)

# 📊 Visualization - Sentiment Distribution
def plot_sentiment_distribution():
    try:
        sentiments = db.collection("sentiment_analysis").stream()
        sentiment_data = [doc.to_dict() for doc in sentiments]
        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            counts = df['label'].value_counts()

            st.subheader("📊 Sentiment Distribution")
            plt.figure(figsize=(6, 4))
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
            st.pyplot(plt)
        else:
            st.warning("⚠️ No sentiment data available for visualization.")
    except Exception as e:
        st.error("❌ Error loading sentiment distribution.")
        st.exception(e)

# 📈 Visualization - Recommendation Accuracy
def plot_recommendation_accuracy():
    try:
        recommendations = db.collection("model_recommendations").stream()
        rec_data = [doc.to_dict() for doc in recommendations]
        if rec_data:
            df = pd.DataFrame(rec_data)
            accuracy = df['is_correct'].value_counts()

            st.subheader("✅ Recommendation Accuracy")
            plt.figure(figsize=(6, 4))
            plt.bar(['Correct', 'Incorrect'], accuracy)
            st.pyplot(plt)
        else:
            st.warning("⚠️ No recommendation data available for visualization.")
    except Exception as e:
        st.error("❌ Error loading recommendation accuracy.")
        st.exception(e)

# 📈 Add Visualization Buttons
st.sidebar.header("📈 Data Visualization")
if st.sidebar.button("📊 Show Sentiment Distribution"):
    plot_sentiment_distribution()

if st.sidebar.button("📈 Show Recommendation Accuracy"):
    plot_recommendation_accuracy()

# 🤖 RAG-based Chatbot Section
st.header("🤖 Financial Insights Chatbot")
chat_query = st.text_input("💬 Ask the chatbot something about the market", key="chat_query")

if st.button("Ask Chatbot"):
    with st.spinner("Generating investment recommendation..."):
        try:
            documents = query_firestore("news")
            if not documents:
                st.warning("⚠️ No news data available for generating a response.")
            else:
                response = generate_rag_response(chat_query, documents)
                st.success(f"📢 **Recommendation:** {response}")
        except Exception as e:
            st.error("❌ Error generating chatbot response.")
            st.exception(e)

# 🗂️ View Stored News Articles
st.sidebar.header("📂 Stored Data")
if st.sidebar.button("📄 Show Stored Articles"):
    with st.spinner("Loading stored articles..."):
        try:
            articles = query_firestore("news")
            if not articles:
                st.warning("⚠️ No articles stored yet.")
            else:
                st.subheader("📰 Stored News Articles")
                for article in articles:
                    st.write(f"**📝 Title:** {article.get('title', 'No Title')}")
                    st.write(f"**📅 Published At:** {article.get('publishedAt', 'Unknown Date')}")
                    st.write(f"**💬 Sentiment:** {article.get('sentiment', 'No Sentiment')}")
                    st.write(f"🔗 **URL:** [Read more]({article.get('url', 'No URL')})")
                    st.markdown("---")
        except Exception as e:
            st.error("❌ Error retrieving stored articles.")
            st.exception(e)

# 🗂️ View Sentiment Analysis Results
st.sidebar.header("📊 Sentiment Analysis Results")
if st.sidebar.button("🧠 Show Sentiment Analysis"):
    try:
        sentiments = db.collection("sentiment_analysis").stream()
        st.subheader("🧠 Sentiment Analysis Results")

        for sentiment in sentiments:
            sentiment_data = sentiment.to_dict()
            st.write(f"**News ID:** {sentiment_data.get('news_id', 'N/A')}")
            st.write(f"**Sentiment:** {sentiment_data.get('label', 'N/A')}")
            st.write(f"**Score:** {sentiment_data.get('score', 'N/A')}")
            st.write(f"**Analyzed At:** {sentiment_data.get('analyzed_at', 'N/A')}")
            st.write("---")

    except Exception as e:
        st.error("❌ Error retrieving sentiment analysis results.")
        st.exception(e)
