import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
