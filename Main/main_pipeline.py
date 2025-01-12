import sys
import os
from dotenv import load_dotenv

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load environment variables
load_dotenv()

# Import agents and Firestore operations
from Agents.news_agent import process_articles
from Agents.economic_data_agent import economic_data_agent
from Agents.sentiment_agent import analyze_sentiment_and_store
from Agents.summarizer_agent import summarize_text
from Agents.table_integration_agent import integrate_tables
from Firebase.firestore_operations import query_firestore

def run_pipeline(keywords, stock_tickers):
    """
    Main pipeline to run the entire workflow:
    1. Fetch news articles.
    2. Fetch economic data.
    3. Perform sentiment analysis.
    4. Integrate data.
    5. Summarize data.
    """
    print("Starting pipeline...")

    # Step 1: Fetch and process news articles
    print("Step 1: Fetching news articles...")
    process_articles(keywords)

    # Step 2: Fetch and store economic data
    print("Step 2: Fetching economic data...")
    economic_data_agent(stock_tickers)

    # Step 3: Perform sentiment analysis
    print("Step 3: Performing sentiment analysis...")
    analyze_sentiment_and_store()

    # Step 4: Integrate tables in Firestore
    print("Step 4: Integrating tables...")
    integrate_tables()

    # Step 5: Summarize the latest data
    print("Step 5: Summarizing data...")
    documents = query_firestore("news")  # Query news collection
    summaries = []
    for doc in documents:
        summary = summarize_text(doc.get("content", ""))
        summaries.append(summary)
    print("Pipeline completed.")
    return summaries

if __name__ == "__main__":
    # Example inputs
    keywords = ["Google", "Tesla", "Microsoft"]
    stock_tickers = ["GOOG", "TSLA", "MSFT"]

    # Run the pipeline
    results = run_pipeline(keywords, stock_tickers)
    print("Summaries:")
    for idx, summary in enumerate(results):
        print(f"{idx+1}. {summary}")
