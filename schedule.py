import os
import json
from typing import List, Dict
from Threads import Threads_scraper
# 新增 firebase-admin
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_functions import firestore_fn
from google.cloud.firestore_v1 import FieldFilter
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
import pytz,logging
from threadsPost import ThreadsAPI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
FIREBASE_KEY_FILE = 'config/firebase_key.json'
class FireStore:
    def __init__(self):
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_KEY_FILE)
            firebase_admin.initialize_app(cred)
        self.client=firestore.client()
        tz = pytz.timezone("Asia/Taipei")
        self.tz=tz
        self.schedule=AsyncIOScheduler(timezone=tz)
        self.threads=ThreadsAPI()
        col_ref=self.client.collection('schedule')
        self.watch = col_ref.on_snapshot(self._on_snapshot)
        self.schedule.start()
    def _expire_schedule(self, doc_id: str):
        """把 status 改成 expired"""
        doc_ref = self.client.collection("schedule").document(doc_id)
        doc_ref.update({"status": "expired"})
        logger.info(f"文件 {doc_id} 已標記為 expired")
    def _publish_and_cleanup(self, text: str, doc_id: str):
        try:
            self.threads.publish_text(text)
            logger.info(f"已發文，ID={doc_id}")
        except Exception as e:
            logger.error(f"發文失敗，ID={doc_id}，錯誤：{e}")
            return
        self._delete_schedule(doc_id)
    def _delete_schedule(self, doc_id: str):
        """真正删除 schedule 集合里指定 ID 的文档"""
        doc_ref = self.client.collection("schedule").document(doc_id)
        if doc_ref.get().exists:
            doc_ref.delete()
            logger.info(f"[{doc_id}] 預排文檔已刪除")
        else:
            logger.warning(f"[{doc_id}] 預排文檔不存在，無需刪除")

    def _get_all_schedule(self):
        rawjobs=self.client.collection('schedule')
        alljobs=[]
        for rowjob in rawjobs.stream():
            job={**rowjob.to_dict(),'id':rowjob.id}
            alljobs.append(job)
        return alljobs
    def _on_snapshot(self, col_snapshot, changes, read_time):
        now = datetime.now(self.tz)
        for change in changes:
            if change.type.name == 'REMOVED':
                continue
            
            doc_snap = change.document
            data     = doc_snap.to_dict()
            doc_id   = doc_snap.id
            status   = data.get('status')
            run_at   = data.get('scheduledTime')
            run_at   = run_at.astimezone(self.tz)
            text     = data.get('content')
            if status == 'expired':
                continue
            elif status == 'pending' and run_at < now:
                self._expire_schedule(doc_id)
                continue
            elif status == 'immediate':
                self._publish_and_cleanup(text, doc_id)
                continue
            # 3. 正常排程
            self.schedule.add_job(
                self._publish_and_cleanup,
                trigger=DateTrigger(run_date=run_at),
                args=[text, doc_id],
                id=doc_id,
                misfire_grace_time=None,
                coalesce=True,
                replace_existing=True
            )
            logger.info(f"[{doc_id}] 已排程至 {run_at.isoformat()}")
    # def initial_add_schedule(self):
    #     now = datetime.now(self.tz)
    #     for job in self._get_all_schedule():
    #         scheduleTime = job["scheduledTime"]
    #         doc_id = job["id"]
    #         text   = job["content"]
    #         status = job.get("status", "")
    #         if status == "pending" and scheduleTime < now:
    #             self._expire_schedule(doc_id)
    #             continue
    #         elif status == 'immidiate':
    #             self._publish_and_cleanup(text=text,doc_id=doc_id)
    #         self.schedule.add_job(
    #             self._publish_and_cleanup,
    #             trigger=DateTrigger(run_date=scheduleTime),
    #             args=[text, doc_id],
    #             id=doc_id,
    #             misfire_grace_time=300
    #         )
    #         logger.info(f"文章已排程，Job ID:{doc_id}，將於 {scheduleTime.isoformat()} 發布")


if __name__ == "__main__":
   ft=FireStore()
   ft.add_schedule()