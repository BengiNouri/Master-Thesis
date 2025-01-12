import firebase_admin
from firebase_admin import credentials, firestore
from utils.logger import get_logger

import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Initialize logger
logger = get_logger(__name__)

# Path to your Firebase service account JSON file
SERVICE_ACCOUNT_PATH = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

def initialize_firebase():
    """
    Initialize Firebase app and return Firestore client.
    """
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase initialized successfully.")
        return db
    except Exception as e:
        logger.error(f"Error initializing Firebase: {e}")
        raise e

def add_document(collection_name, document_data):
    """
    Add a document to Firestore.

    Args:
        collection_name (str): Firestore collection name.
        document_data (dict): Data to save.
    """
    try:
        db = firestore.client()
        db.collection(collection_name).add(document_data)
        logger.info(f"Document added to Firestore in {collection_name} collection.")
    except Exception as e:
        logger.error(f"Error adding document to Firestore: {e}")
        raise e

def query_firestore(collection_name):
    """
    Query documents from a Firestore collection.

    Args:
        collection_name (str): Firestore collection name.

    Returns:
        list: List of documents.
    """
    try:
        db = firestore.client()
        docs = db.collection(collection_name).stream()
        results = [doc.to_dict() for doc in docs]
        logger.info(f"Queried {len(results)} documents from Firestore collection: {collection_name}")
        return results
    except Exception as e:
        logger.error(f"Error querying Firestore: {e}")
        raise e

if __name__ == "__main__":
    db = initialize_firebase()
