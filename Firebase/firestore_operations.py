import firebase_admin
from firebase_admin import credentials, firestore
from utils.logger import get_logger

import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Logger initialization
logger = get_logger(__name__)

# Path to your service account JSON file
SERVICE_ACCOUNT_PATH = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

def initialize_firestore():
    """
    Initialize Firebase and Firestore client.

    Returns:
        Firestore client instance.
    """
    try:
        if not firebase_admin._apps:
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
        collection_name (str): Name of the Firestore collection.
        document_data (dict): Data to add as a document.
    """
    try:
        db = initialize_firestore()
        if not isinstance(document_data, dict):
            raise ValueError("Document data must be a dictionary.")
        
        db.collection(collection_name).add(document_data)
        logger.info(f"Document added to Firestore in {collection_name} collection.")
    except Exception as e:
        logger.error(f"Error adding document to Firestore: {e}")
        raise e

def query_firestore(collection_name):
    """
    Query documents from Firestore.

    Args:
        collection_name (str): Name of the Firestore collection.

    Returns:
        list: List of queried documents.
    """
    try:
        db = initialize_firestore()
        docs = db.collection(collection_name).stream()
        results = [doc.to_dict() for doc in docs]
        logger.info(f"Queried {len(results)} documents from Firestore collection: {collection_name}")
        return results
    except Exception as e:
        logger.error(f"Error querying Firestore: {e}")
        raise e

# Test the Firestore operations
if __name__ == "__main__":
    try:
        test_data = {"title": "Sample Article", "content": "This is a sample content."}
        add_document("test", test_data)
        documents = query_firestore("test")
        print(documents)
    except Exception as e:
        print(f"Error: {e}")
