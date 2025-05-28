import os
from dotenv import load_dotenv

load_dotenv()
THREADS_ACCESS_TOKEN=os.environ.get("THREADS_ACCESS_TOKEN")
THREADS_ID=os.environ.get("THREADS_ID")
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")
print("環境變數載入完成")