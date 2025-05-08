# export_firestore.py

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ Load .env & initialize Firestore
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

def init_db():
    cred_paths = [
        r"C:\MasterThesis\Keys.json",
        r"C:\Users\sajad\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
        r"C:\Users\Benja\OneDrive\Skole\DevRepos\Master Thesis\Keys.json",
    ]
    for p in cred_paths:
        if os.path.exists(p) and not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(p))
            break
    return firestore.client()

db = init_db()

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣ Helper to export one collection
# ─────────────────────────────────────────────────────────────────────────────
def export_collection(name: str, out_dir: str):
    snaps = list(db.collection(name).stream())
    data = [doc.to_dict() for doc in snaps]
    count = len(data)

    # JSON
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, indent=2, ensure_ascii=False)
    print(f"✅ Exported {count} docs to {json_path}")

    # CSV (if non‐empty)
    if count:
        df = pd.DataFrame(data)
        csv_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Exported {count} docs to {csv_path}")
    else:
        print(f"ℹ️ Collection '{name}' is empty — skipped CSV")

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣ Main: run on each collection, save into this script’s folder
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # the directory where this script lives
    out_dir = os.path.dirname(os.path.abspath(__file__))

    collections = [
        "news",
        "model_recommendations",
        "latest_economic_data",
        "economic_data"
    ]

    for col in collections:
        export_collection(col, out_dir)
