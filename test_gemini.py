import os
from dotenv import load_dotenv
from google import genai

# 這行會讀取 .env 檔案並載入到環境變數中
load_dotenv()

# 這裡明確指定從環境變數讀取，或者讓 Client() 自動抓取 GOOGLE_API_KEY
client = genai.Client()
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents="Explain how AI works in a few words"
    )
    print(response.text)
except Exception as e:
    print(f"Error: {e}")