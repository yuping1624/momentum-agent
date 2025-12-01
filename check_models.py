import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# If .env cannot be read, paste the key directly in the quotes below
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Querying available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error occurred: {e}")
