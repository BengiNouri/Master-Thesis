from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access your keys using os.getenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")


def clean_text(text):
    if not text:
        return ""
    return text.strip().replace("\n", " ").lower()
