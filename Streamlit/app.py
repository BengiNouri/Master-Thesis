import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit as st

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import custom modules (updated paths to align with previous work)
from Agents.news_agent import process_articles  # Handles fetching and storing articles
from Agents.rag_agent import generate_rag_response
from Firebase.firestore_operations import query_firestore

# Streamlit app configuration
st.set_page_config(page_title="Financial Insights Dashboard", layout="wide")
st.title("📊 Financial Insights Dashboard")

# Sidebar Input Section
st.sidebar.header("Search Options")
query = st.sidebar.text_input("🔍 Search Articles", "Tesla")
start_date = st.sidebar.date_input("📅 Start Date", datetime.today() - timedelta(days=30))
end_date = st.sidebar.date_input("📅 End Date", datetime.today())
max_articles = st.sidebar.number_input("📄 Max Articles", min_value=1, max_value=50, value=10)

# 🔎 Fetch News and Process
if st.sidebar.button("Fetch News"):
    st.subheader("📰 Fetching News Articles")
    try:
        # Fetch, store, and analyze articles (linked with sentiment and economic data)
        process_articles([query], max_articles)
        st.success("✅ News articles fetched, analyzed, and stored.")
    except Exception as e:
        st.error("❌ Error fetching and processing articles.")
        st.error(f"{e}")

# 🤖 RAG-based Chatbot
st.header("🤖 Financial Insights Chatbot")
chat_query = st.text_input("💬 Ask the chatbot", key="chat_query")

if st.button("Ask Chatbot"):
    try:
        # Retrieve all news articles for RAG context
        documents = query_firestore("news")
        if not documents:
            st.warning("⚠️ No documents available for generating a response.")
        else:
            # Generate context-aware response using RAG
            response = generate_rag_response(chat_query, documents)
            st.success(f"🤔 Response: {response}")
    except Exception as e:
        st.error("❌ Error generating chatbot response.")
        st.error(f"{e}")

# 🗂️ View Stored Articles
st.sidebar.header("📰 Stored Articles")
if st.sidebar.button("Show Stored Articles"):
    try:
        articles = query_firestore("news")
        if not articles:
            st.warning("⚠️ No articles stored yet.")
        else:
            st.subheader("📰 Stored News Articles")
            for article in articles:
                st.write(f"**Title:** {article.get('title', 'No Title')}")
                st.write(f"**Published At:** {article.get('publishedAt', 'Unknown Date')}")
                st.write(f"**URL:** [Link]({article.get('url', 'No URL')})")
                st.write("---")
    except Exception as e:
        st.error("❌ Error retrieving stored articles.")
        st.error(f"{e}")
