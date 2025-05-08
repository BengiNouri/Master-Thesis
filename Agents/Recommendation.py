# recommendation_generation.py

import firebase_admin
from firebase_admin import credentials, firestore
import os

def initialize_firebase():
    """
    Initialize Firebase with fallback credential paths and return a Firestore client.
    """
    # Define your credential paths
    vm_path = r"C:\MasterThesis\Keys.json"
    primary_path = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
    fallback_path = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

    if not firebase_admin._apps:
        cred = None
        if os.path.exists(vm_path):
            cred = credentials.Certificate(vm_path)
        elif os.path.exists(primary_path):
            cred = credentials.Certificate(primary_path)
        elif os.path.exists(fallback_path):
            cred = credentials.Certificate(fallback_path)
        else:
            raise FileNotFoundError("Firebase credentials file not found in any of the specified paths.")
        
        firebase_admin.initialize_app(cred)

    return firestore.client()

db = initialize_firebase()

# Define stock mapping (could also be loaded from Firestore)
STOCK_MAPPING = {
    "tesla": "TSLA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "novo nordisk": "NVO",
    # Add more mappings as needed
}

def fetch_recommendation(stock_ticker):
    """
    Fetch the stored recommendation and sentiment summary for a given stock ticker.
    """
    try:
        rec_doc = db.collection("recommendations").document(stock_ticker).get()
        if not rec_doc.exists:
            print(f"⚠️ No recommendation found for {stock_ticker}.")
            return None
        return rec_doc.to_dict()
    except Exception as e:
        print(f"❌ Error fetching recommendation for {stock_ticker}: {e}")
        return None

def fetch_closing_prices(stock_ticker):
    """
    Fetch the latest and previous closing prices for a given stock ticker.
    Assumes there's a 'stock_prices' collection with documents named by stock_ticker.
    """
    try:
        price_doc = db.collection("stock_prices").document(stock_ticker).get()
        if not price_doc.exists:
            print(f"⚠️ No price data found for {stock_ticker}.")
            return None, None

        price_data = price_doc.to_dict()
        latest_close = price_data.get("latest_close")
        previous_close = price_data.get("previous_close")
        return latest_close, previous_close
    except Exception as e:
        print(f"❌ Error fetching stock prices for {stock_ticker}: {e}")
        return None, None

def evaluate_recommendation(gpt_rec, latest_close, previous_close):
    """
    Determine if the recommendation aligns with the actual stock movement.
    """
    if not latest_close or not previous_close:
        return False

    try:
        if gpt_rec.lower() in ["buy", "hold"] and latest_close > previous_close:
            return True
        elif gpt_rec.lower() == "sell" and latest_close < previous_close:
            return True
        else:
            return False
    except Exception as e:
        print(f"❌ Error evaluating recommendation: {e}")
        return False

def store_evaluation(stock_ticker, is_correct):
    """
    Store the evaluation result in Firestore under the recommendations document.
    """
    try:
        rec_doc_ref = db.collection("recommendations").document(stock_ticker)
        rec_doc_ref.update({
            "is_correct": is_correct
        })
        print(f"✅ Stored evaluation for {stock_ticker}: Correct? {is_correct}")
    except Exception as e:
        print(f"❌ Error storing evaluation for {stock_ticker}: {e}")

def generate_report(stock_ticker, recommendation_data, is_correct):
    """
    Generate and print a report based on the recommendation and evaluation.
    """
    if not recommendation_data:
        print(f"⚠️ No data to generate report for {stock_ticker}.")
        return

    aggregator_rec = recommendation_data.get("aggregator_rec")
    gpt_rec = recommendation_data.get("gpt_rec")
    sentiment_sum = recommendation_data.get("sentiment_sum", {})
    
    print(f"\n📄 Report for {stock_ticker}:")
    print(f"Aggregator Recommendation: {aggregator_rec}")
    print(f"GPT Recommendation: {gpt_rec}")
    print(f"Sentiment Summary: {sentiment_sum}")
    print(f"Recommendation Correct? {'✅ Yes' if is_correct else '❌ No'}")

def main():
    # Iterate over each stock in the mapping
    for stock_name, stock_ticker in STOCK_MAPPING.items():
        print(f"\n🔍 Processing {stock_ticker}...")

        # Fetch stored recommendation
        recommendation_data = fetch_recommendation(stock_ticker)
        if not recommendation_data:
            continue

        aggregator_rec = recommendation_data.get("aggregator_rec")
        gpt_rec = recommendation_data.get("gpt_rec")
        sentiment_sum = recommendation_data.get("sentiment_sum")

        # Fetch stock prices for evaluation
        latest_close, previous_close = fetch_closing_prices(stock_ticker)

        # Evaluate recommendation correctness
        is_correct = evaluate_recommendation(gpt_rec, latest_close, previous_close)

        # Store the evaluation result
        store_evaluation(stock_ticker, is_correct)

        # Generate a report
        generate_report(stock_ticker, recommendation_data, is_correct)

    print("\n✅ Recommendation Generation Workflow completed successfully!")

if __name__ == "__main__":
    main()
