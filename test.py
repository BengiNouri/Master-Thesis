import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello! What's the weather today?"}
    ]
)

# Instead of response['choices'][0]['message']['content'] ...
# ... do:
print(response.choices[0].message.content)
