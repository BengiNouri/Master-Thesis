# Master-Thesis

This repository contains the core scripts and resources for a **Master’s Thesis** project focused on real-time sentiment analysis and LLM-driven investment recommendations. The pipeline integrates multiple APIs, a Firebase database, and locally stored credentials to generate daily **Buy/Hold/Sell** signals for selected stocks.

## Table of Contents

1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Setup](#setup)  
   3.1. [Clone the Repository](#clone-the-repository)  
   3.2. [Install Dependencies](#install-dependencies)  
   3.3. [Environment Variables](#environment-variables)  
   3.4. [Firebase Credentials](#firebase-credentials)  
4. [Running the Pipeline](#running-the-pipeline)  
5. [Additional Notes](#additional-notes)  
6. [Contact](#contact)

---

## Overview

**Objective**  
The main goal of this thesis is to investigate whether **NLP-based sentiment analysis** and **LLM-powered** recommendations can offer meaningful insights into short-term stock price movements.

**Key Features**  
- **Daily Pipeline**: Scripts fetch real-time financial news, apply **FinBERT** sentiment analysis, and funnel aggregated results into **GPT-4o-mini** for final recommendations.  
- **Firebase Integration**: Articles, sentiment labels, and recommendations are stored and linked in a **Firestore** database.  
- **Modular Agents**: Each agent handles a discrete function, such as news retrieval, sentiment analysis, or final advisory logic.

## Project Structure

Master-Thesis/ ├── Agents/ │ ├── news_agent.py # Fetches articles from NewsAPI, links them to Firestore │ ├── sentiment_agent.py # FinBERT sentiment classification │ ├── rag_agent.py # GPT-4o-mini + aggregator for Buy/Hold/Sell │ ├── economic_data_agent.py # Fetches fundamentals from yfinance │ └── ... ├── Data Analysis/ # (Optional) Jupyter notebooks or scripts for data exploration ├── Firebase/ # Firebase config or utility scripts ├── Streamlit/ # (Optional) UI or dashboard scripts ├── daily_run.py # Main orchestrator (daily pipeline) ├── config.py # Common config settings ├── requirements.txt # Python dependencies ├── README.md # This file! └── ...


---

## Setup

### 3.1. Clone the Repository

```bash
git clone https://github.com/BengiNouri/Master-Thesis.git
cd Master-Thesis
```

### 3.2 Install Dependencies
Make sure you have Python 3.9+ installed. Then run:

pip install --upgrade pip
pip install -r requirements.txt

### 3.3 Environment Variables

Create an .env file in the project’s root directory. For example:

OPENAI_API_KEY=your_openai_api_key
NEWS_API_KEY=your_news_api_key

These environment variables are used by various scripts (e.g., rag_agent.py, news_agent.py) to authenticate with OpenAI and NewsAPI.

    Note: Ensure that your .env file is excluded in your .gitignore so as not to commit any sensitive data.

### 3.4. Firebase Credentials

    Obtain your Firebase service account JSON key.
    Place it in the directory your scripts expect (e.g., Firebase/Keys.json).
    Confirm the path in scripts like economic_data_agent.py matches your actual JSON file location.

    Important: Keep your key JSON file out of public repos to avoid potential credential exposure.

# Running the Pipeline

Once everything is installed and configured:

python daily_run.py

This workflow will:

    Fetch the latest financial news for each stock in stock_tickers.
    Store and link the articles in Firestore, associating them with the relevant ticker.
    Perform FinBERT sentiment analysis, labeling each article as Positive, Neutral, or Negative.
    Pass aggregated sentiment to GPT-4o-mini, which issues a Buy/Hold/Sell recommendation.
    Compare the recommendation to actual price movements (via yfinance), storing results back in Firestore.

# Additional Notes

    Rate Limiting
    If you encounter Too Many Requests errors from yfinance, insert short time.sleep() calls in the main loop (e.g., daily_run.py) to avoid consecutive rapid requests.

    Testing
    If a test.py or notebooks in Data Analysis/ exist, you can selectively run or test portions of the pipeline.

    Future Enhancements
        Detect and remove duplicate news articles.
        Use additional filters for “noisy” or biased data sources.
        Experiment with alternate LLMs or advanced retrieval methods.



# Contact

For questions, comments, or collaboration requests:

    Name: Benjamin Sajad Nouri
    Email: 201810726@post.au.dk
