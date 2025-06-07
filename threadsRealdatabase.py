import os
import json
from typing import List, Dict
from Threads import Threads_scraper
# 新增 firebase-admin
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter
from datetime import datetime, timezone, timedelta
# --------------------------------------------
# 1. 初始化 Firestore（只做一次）
# --------------------------------------------
# 把你在 Firebase 控制台下載的 service-account key 放在同目錄下
FIREBASE_KEY_FILE = 'config/firebase_key.json'

def init_firebase():
    # 如果已經初始化就直接拿 client
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_KEY_FILE)
        firebase_admin.initialize_app(cred)
    return firestore.client()

# --------------------------------------------
# 2. 將貼文 list 寫進 Firestore
# --------------------------------------------
def store_posts_to_firestore(posts: List[Dict]):
    db = init_firebase()
    col = db.collection('schedule_job')  # 你要的 collection 名稱
    for post in posts:
        # 我們用貼文 id 當作 Document ID，這樣不會重複
        doc_id = post.get("id")
        if not doc_id:
            continue
        print(post)
        # 如果你要額外加一個 timestamp，可以這樣做：
        data = {
            "username":    post.get("username", ""),
            "text":        post.get("text", ""),
            "like_count":  post.get("like_count", 0),
            "reply_count": post.get("reply_count", 0),
            "timestamp":   datetime.fromisoformat(post.get("timestamp", 0)), # 這是原始的時間戳
            "Emotion":     post.get("Emotion",0),
            "Trend":       post.get("Trend",0),
            "Practical":   post.get("Practical",0),
            "Identity":    post.get("Identity",0),
            # "fetched_at":  firestore.SERVER_TIMESTAMP,  # Firestore 內建時間戳
        }
        # set() 會覆蓋同 ID 的舊資料；用 update() 可以只改部分欄位
        col.document(doc_id).set(data)

    print(f"✅ 已將 {len(posts)} 筆貼文寫入 Firestore。")
def fetch_top_query(limit:int=50,label:str="Emotion",days_keep:int=7):
    db= init_firebase()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_keep)
    col = db.collection('threads_posts')\
        .where(filter=FieldFilter("timestamp",">=",cutoff))\
        .where(filter=FieldFilter(label, "==", True))\
        .order_by('like_count', direction=firestore.Query.DESCENDING)\
        .limit(limit)
    docs = col.stream()
    posts =[doc.to_dict() for doc in docs]
    return posts
def delete_posts(label:str="ALL",days_keep:int=7):
    db= init_firebase()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_keep)
    if label=="ALL":
        col = db.collection('threads_posts')\
            .where(filter=FieldFilter("timestamp","<=",cutoff))
    else:
        col = db.collection('threads_posts')\
            .where(filter=FieldFilter(label, "==", True))\
            .where(filter=FieldFilter("timestamp","<=",cutoff))
    docs =list(col.stream())
    batch=db.batch()
    for doc in docs:
        batch.delete(doc.reference)
    batch.commit()
    print(f"✅ 已刪除 {len(docs)} 筆貼文。")
if __name__ == "__main__":
    posts=fetch_top_query(limit=10,label="Trend",days_keep=20)
    delete_posts(label="Emotion",days_keep=0)
