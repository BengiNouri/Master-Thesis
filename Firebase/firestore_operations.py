import firebase_admin
from firebase_admin import credentials, firestore
from utils.logger import get_logger
import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Logger initialization
logger = get_logger(__name__)

# ✅ Paths for service account credentials
VM_PATH = r"C:\MasterThesis\Keys.json"
PRIMARY_PATH = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
FALLBACK_PATH = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

# 🔥 Singleton Firestore Client
db = None

def initialize_firestore():
    """
    Initialize Firebase and Firestore client (singleton).
    Returns:
        Firestore client instance.
    """
    global db
    try:
        if db is None:
            if not firebase_admin._apps:
                # Dynamically choose the correct path
                if os.path.exists(VM_PATH):
                    cred_path = VM_PATH
                elif os.path.exists(PRIMARY_PATH):
                    cred_path = PRIMARY_PATH
                elif os.path.exists(FALLBACK_PATH):
                    cred_path = FALLBACK_PATH
                else:
                    raise FileNotFoundError("Firebase credentials not found in any path.")
                
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                logger.info(f"Firebase initialized with credentials from: {cred_path}")

            db = firestore.client()
            logger.info("Firestore client initialized successfully.")
        return db

    except Exception as e:
        logger.error(f"Error initializing Firebase: {e}")
        raise e

def add_document(collection_name, document_data):
    """
    Add a document to Firestore.

    Args:
        collection_name (str): Firestore collection name.
        document_data (dict): Data to store in the document.

    Returns:
        str: Document ID of the newly added document.
    """
    try:
        db = initialize_firestore()
        if not isinstance(document_data, dict):
            raise ValueError("Document data must be a dictionary.")

        doc_ref = db.collection(collection_name).add(document_data)
        doc_id = doc_ref[1].id  # Extract document ID
        logger.info(f"Document added to '{collection_name}' with ID: {doc_id}")
        return doc_id

    except Exception as e:
        logger.error(f"Error adding document to Firestore: {e}")
        raise e

def query_firestore(collection_name):
    """
    Query documents from Firestore.

    Args:
        collection_name (str): Firestore collection name.

    Returns:
        list: List of queried documents as dictionaries.
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

# 🧪 Test Firestore Operations
if __name__ == "__main__":
    try:
        test_data = {
            "title": "Sample Article",
            "content": "This is a sample content.",
            "timestamp": firestore.SERVER_TIMESTAMP
        }

        # Test adding a document
        doc_id = add_document("test", test_data)
        print(f"Document added with ID: {doc_id}")

        # Test querying documents
        documents = query_firestore("test")
        print("Queried Documents:", documents)

    except Exception as e:
        print(f"❌ Error during Firestore operations: {e}")
