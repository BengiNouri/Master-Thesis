# Master Thesis

This repository contains the full codebase and resources for a **Master’s Thesis** project focused on real-time sentiment analysis and LLM-driven stock recommendations. The pipeline integrates multiple APIs, a Firebase database, and a modular agent architecture to generate daily **Buy / Hold / Sell** signals for selected stocks.

## Table of Contents

1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Setup](#setup)  
   - [Clone the Repository](#clone-the-repository)  
   - [Install Dependencies](#install-dependencies)  
   - [Environment Variables](#environment-variables)  
   - [Firebase Credentials](#firebase-credentials)  
4. [Running the Pipeline](#running-the-pipeline)  
5. [Additional Notes](#additional-notes)  
6. [Contact](#contact)

---

## Overview

### 🌟 Objective  
The goal of this thesis is to evaluate whether **NLP-based sentiment signals** and **LLM-powered decision-making** can yield actionable insights for short-term stock forecasting.

### 🔍 Key Features  
- **Daily Pipeline**: Scrapes financial news, applies **FinBERT** for sentiment scoring, and queries **GPT-4o-mini** to issue stock-level investment recommendations.  
- **Firebase Integration**: Articles, sentiment scores, and decisions are stored in **Firestore** for traceability and analysis.  
- **Modular Architecture**: Individual agents handle data collection, sentiment classification, advisory logic, and economic data ingestion.

---

## Project Structure

```
Master-Thesis/
├── Agents/
│   ├── news_agent.py              # Fetches articles via NewsAPI and links to Firestore
│   ├── sentiment_agent.py         # Performs FinBERT-based sentiment classification
│   ├── rag_agent.py               # Aggregates sentiment + GPT-4o-mini advisory logic
│   ├── economic_data_agent.py     # Retrieves market metrics from yfinance
│   └── ...
├── Data Analysis/                 # Optional: notebooks and scripts for exploration
├── Firebase/                      # Firebase config and credentials
├── Streamlit/                     # Optional UI (Streamlit dashboard)
├── daily_run.py                   # Main orchestrator for daily execution
├── config.py                      # Shared configuration
├── requirements.txt               # Python dependencies
└── README.md                      # You're reading it!
```

---

## Setup

### 3.1. Clone the Repository

```bash
git clone https://github.com/BengiNouri/Master-Thesis.git
cd Master-Thesis
```

### 3.2. Install Dependencies

Ensure Python 3.9+ is installed. Then run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. Environment Variables

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key
NEWS_API_KEY=your_news_api_key
```

> **Note**: Make sure your `.env` is listed in `.gitignore` to avoid exposing credentials.

### 3.4. Firebase Credentials  🚀 NEW

1.  **Download** your Firebase service-account key *(JSON format)* from
    the Firebase console.

2.  Place it **anywhere you like** and tell the code where to find it by
    adding this line to your `.env` (recommended) **or** by exporting the
    variable in your shell:

    ```env
    FIREBASE_KEY_PATH=/full/path/to/Keys.json
    ```

    ```bash
    # Linux / macOS
    export FIREBASE_KEY_PATH=~/Secrets/Keys.json
    # Windows (PowerShell)
    setx FIREBASE_KEY_PATH "C:\Secrets\Keys.json"
    ```

3.  If `FIREBASE_KEY_PATH` is **not** set, the helper
    `utils/firebase_utils.initialize_firebase()` falls back to
    `Firebase/Keys.json` relative to the repo root.  
    (Rename or move the file if you prefer that default.)

> 🔐 **Security Tip**  
> Keep the key out of version control. `.gitignore` already excludes
> `*.json` under `Firebase/`, but double-check before pushing.

The same variable is honoured by **all scripts** (`daily_run.py`,
`news_agent.py`, `sentiment_agent.py`, Streamlit dashboards, etc.), so
you only configure the path once.

---

## Running the Pipeline

To execute the full end-to-end pipeline:

```bash
python daily_run.py
```

The pipeline will:

- Fetch news articles via NewsAPI for the specified tickers
- Store and tag articles in Firestore
- Perform sentiment analysis using FinBERT
- Aggregate sentiment and generate recommendations via GPT-4o-mini
- Compare predictions with actual price movements
- Log outcomes in Firestore for evaluation

---

## Additional Notes

### ⏱️ Rate Limiting
If you encounter `TooManyRequests` errors from Yahoo Finance, add a short `time.sleep()` delay between API calls in `daily_run.py`.

### 🧪 Testing
You can test pipeline components in isolation using:
- `test.py` (if provided)
- Notebooks under `Data Analysis/` for ad hoc experiments

### 🚀 Future Enhancements
- Add deduplication for news articles  
- Implement source credibility scoring  
- Experiment with alternate LLMs or RAG configurations  
- Integrate explainability overlays (e.g., SHAP)  
- Add real-time streaming support

---

## Contact

Feel free to reach out for collaboration or questions:

**Benjamin Sajad Nouri**  
📧 [benjamin_nouri@outlook.dk](mailto:benjamin_nouri@outlook.dk)  
📘 MSc in Business Intelligence – Aarhus BSS, Aarhus University
