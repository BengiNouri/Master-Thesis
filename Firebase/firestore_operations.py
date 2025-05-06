import os
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from utils.logger import get_logger

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()
logger = get_logger(__name__)

# Paths for service account credentials
VM_PATH = r"C:\MasterThesis\Keys.json"
PRIMARY_PATH = r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"
FALLBACK_PATH = r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json"

# Singleton Firestore client
_db: Optional[firestore.Client] = None

# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NewsArticle:
    id: str
    title: str
    content: str
    url: str
    publishedAt: str
    source: str
    keywords: List[str]
    economic_data_id: str
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]
    analyzed_at: Optional[str]

@dataclass
class Recommendation:
    id: str
    aggregator_rec: str
    gpt_rec: str
    sentiment_sum: Dict[str, float]
    timestamp: Any

@dataclass
class SentimentRecord:
    id: str
    news_id: str
    label: str
    score: float
    analyzed_at: str

@dataclass
class EconomicData:
    id: str
    stock_ticker: str
    long_name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    current_price: Optional[float]
    previous_close: Optional[float]
    market_cap: Optional[float]
    volume: Optional[int]
    week52_high: Optional[float]
    week52_low: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    fetched_at: Optional[str]
    status: str
    error_msg: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# Firestore Initialization
# ─────────────────────────────────────────────────────────────────────────────
def initialize_firestore() -> firestore.Client:
    global _db
    if _db is None:
        if not firebase_admin._apps:
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
            logger.info(f"Initialized Firebase with credentials from {cred_path}")
        _db = firestore.client()  # type: ignore
        logger.info("Firestore client ready.")
    return _db  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Generic Add Document
# ─────────────────────────────────────────────────────────────────────────────
def add_document(collection: str, document_data: Dict[str, Any]) -> str:
    if not isinstance(document_data, dict):
        raise ValueError("Document data must be a dict.")
    db = initialize_firestore()
    doc_ref = db.collection(collection).add(document_data)
    doc_id = doc_ref[1].id
    logger.info(f"Added document to {collection} with ID {doc_id}")
    return doc_id

# ─────────────────────────────────────────────────────────────────────────────
# Generic Query Helper
# ─────────────────────────────────────────────────────────────────────────────
def query_collection(
    collection: str,
    where_clauses: Optional[List[tuple]] = None,
    order_by: Optional[tuple] = None,
    limit: Optional[int] = None
) -> List[firestore.DocumentSnapshot]:
    db = initialize_firestore()
    col = db.collection(collection)
    if where_clauses:
        for field, op, value in where_clauses:
            col = col.where(field, op, value)
    if order_by:
        col = col.order_by(order_by[0], direction=order_by[1])
    if limit:
        col = col.limit(limit)
    try:
        return list(col.stream())
    except ResourceExhausted as e:
        logger.error(f"Quota exceeded querying {collection}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error querying {collection}: {e}")
        return []

# ─────────────────────────────────────────────────────────────────────────────
# Specialized Queries with Fallback Logic
# ─────────────────────────────────────────────────────────────────────────────
def query_news_articles(
    ticker: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100
) -> List[NewsArticle]:
    clauses = []
    if ticker:
        clauses.append(("economic_data_id", "==", ticker))
    if start and end:
        clauses.append(("publishedAt", ">=", start))
        clauses.append(("publishedAt", "<=", end))

    snaps = query_collection(
        "news", where_clauses=clauses,
        order_by=("publishedAt", firestore.Query.DESCENDING),
        limit=limit
    )
    # Fallback if empty (composite index missing)
    if not snaps and ticker and start and end:
        raw_snaps = query_collection(
            "news",
            where_clauses=[("economic_data_id", "==", ticker)],
            order_by=("publishedAt", firestore.Query.DESCENDING),
            limit=limit * 2
        )
        snaps = [
            snap for snap in raw_snaps
            if start <= snap.to_dict().get("publishedAt", "") <= end
        ][:limit]

    articles: List[NewsArticle] = []
    for snap in snaps:
        d = snap.to_dict()
        articles.append(NewsArticle(
            id=snap.id,
            title=d.get("title", ""),
            content=d.get("content", ""),
            url=d.get("url", ""),
            publishedAt=d.get("publishedAt", ""),
            source=d.get("source", ""),
            keywords=d.get("keywords", []),
            economic_data_id=d.get("economic_data_id", ""),
            sentiment_label=d.get("sentiment_label"),
            sentiment_score=d.get("sentiment_score"),
            analyzed_at=d.get("analyzed_at")
        ))
    return articles


def query_recommendations(limit: int = 100) -> List[Recommendation]:
    snaps = query_collection(
        "recommendations",
        order_by=("timestamp", firestore.Query.DESCENDING),
        limit=limit
    )
    recs: List[Recommendation] = []
    for snap in snaps:
        d = snap.to_dict()
        recs.append(Recommendation(
            id=snap.id,
            aggregator_rec=d.get("aggregator_rec", ""),
            gpt_rec=d.get("gpt_rec", ""),
            sentiment_sum=d.get("sentiment_sum", {}),
            timestamp=d.get("timestamp")
        ))
    return recs


def query_sentiment_analysis(limit: int = 100) -> List[SentimentRecord]:
    snaps = query_collection(
        "sentiment_analysis",
        order_by=("analyzed_at", firestore.Query.DESCENDING),
        limit=limit
    )
    records: List[SentimentRecord] = []
    for snap in snaps:
        d = snap.to_dict()
        records.append(SentimentRecord(
            id=snap.id,
            news_id=d.get("news_id", ""),
            label=d.get("label", ""),
            score=d.get("score", 0.0),
            analyzed_at=d.get("analyzed_at", "")
        ))
    return records

# ─────────────────────────────────────────────────────────────────────────────
# Raw Query Fallback
# ─────────────────────────────────────────────────────────────────────────────
def query_firestore(collection: str) -> List[Dict[str, Any]]:
    snaps = query_collection(collection)
    return [snap.to_dict() for snap in snaps]

# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(query_news_articles(ticker="TSLA", limit=3))
    print(query_recommendations(limit=3))
    print(query_sentiment_analysis(limit=3))
