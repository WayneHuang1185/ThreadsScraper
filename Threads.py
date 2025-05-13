"""Threads Scraper (with total post count)

支援：
* 單一帳號或多帳號批次
* --json 輸出原始 JSON
* 額外顯示每個帳號的 **總貼文數**（若能從 JSON 取得）
"""

import asyncio
import json
from typing import List, Dict, Optional, Tuple
import jmespath
from nested_lookup import nested_lookup
from parsel import Selector
from playwright.async_api import async_playwright
from datetime import datetime,timedelta, timezone
import os
class Threads_scraper:
        def __init__(self,search_choice:str="like_count",acc:bool=False,username:Optional[List[str]]=None):
            try:
                with open('existID.json', 'r', encoding='utf-8') as f:
                    self.seen = set(json.load(f))
            except (FileNotFoundError, json.JSONDecodeError):
                    self.seen = set()
            self.url="https://www.threads.net/@"
            self.QUERY_SELECTOR = "script[type='application/json'][data-sjs]"
            self.search_choice=search_choice
            self.acc=acc
            self.gclike=0 #greater like counts
            self.gcreply=0 #greater reply counts
            self.lttext=0 #less than text counts
            self.image_retrieve=False
            self.username=username
        def filter_setting(self,gclike:int=0, gcreply:int=0, lttext:int=0, image_retrieve:bool=False):
            self.gclike=gclike if gclike>0 else 0
            self.gcreply=gcreply if gcreply>0 else 0
            self.lttext=lttext if lttext>0 else 0
            self.image_retrieve= True  if image_retrieve else False
        def _parse_post(self,item: Dict) -> Dict:
            """把原始 thread_items 物件濃縮成精簡欄位。"""
            return jmespath.search(
                """
                {
                id: post.id,
                code: post.code,
                text: post.caption.text,
                like_count: post.like_count,
                reply_count: post.text_post_app_info.direct_reply_count,
                username: post.user.username,
                timestamp: post.taken_at
                }
                """,
                item,
            )
        def _sort_posts(self,posts: List[Dict], sort_key: str, ascending: bool = False) -> List[Dict]:
            """根據 sort_key 對貼文列表排序，預設 by sort_key 降冪。"""
            # 如果 key 不存在，預設值為 0 或空字串
            return sorted(
                posts,
                key=lambda p: p.get(sort_key) or 0,
                reverse=not ascending,
        )
        async def Top_crawl(self,batch:int=3)-> List[Dict]:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                usertmp=self.username
                posts: List[Dict] = []
                for idx,username in enumerate(self.username):
                    url=self.url+username
                    await page.goto(url, timeout=60000)
                    # 等待隱藏 JSON 掛進 DOM
                    await page.wait_for_selector(self.QUERY_SELECTOR, state="attached", timeout=60000)
                    selector = Selector(await page.content())
                    scripts = selector.css(f"{self.QUERY_SELECTOR}::text").getall()
                    done=False
                    pre=len(posts)
                    for payload in scripts:
                        data = json.loads(payload)
                        # 第一次嘗試抽出總貼文數（JSON 中可能出現 user.media_count 或 thread_count）
                        if '"thread_items"' not in payload:
                            continue
                        for group in nested_lookup("thread_items", data):
                            for item in group:
                                parsed = self._parse_post(item)
                                if self.gclike and int(parsed.get("like_count", 0)) < self.gclike:
                                    continue
                                if self.gcreply and int(parsed.get("reply_count", 0)) < self.gcreply:
                                    continue
                                if self.lttext and len(parsed.get("text", "") or "") > self.lttext:
                                    continue
                                if self.image_retrieve == parsed.get("media_urls"):
                                    continue
                                #print(f"post id: {parsed['id']}")
                                if parsed["id"] not in self.seen:
                                    posts.append(parsed)
                                    # print(self.seen)
                                    if len(posts) >= (idx+1)*batch: 
                                        break
                            if len(posts) >= (idx+1)*batch: 
                                done=True
                                break
                        if done:
                            break
                    if len(posts) == pre:
                        usertmp.remove(username)
                self.username=usertmp
                posts=self._sort_posts(posts,self.search_choice,self.acc)
                return posts    
        def printPost(self,posts):
            for idx, post in enumerate(posts, 1):
                excerpt = (post["text"] or "").replace("\n", " ")[:150]
                url = f"https://www.threads.net/@{post['username']}/post/{post['code']}"
                print(f"{idx}. {excerpt}…")
                print(f"   👍 {post['like_count']}   💬 {post['reply_count']}   {url}\n")
        def printJosn(self,posts):
            out={"post":posts}
            print(json.dumps(out, ensure_ascii=False, indent=1))
        def getJosn(self,posts):
            """將貼文轉換為 JSON 格式"""
            cleaned_posts = []
            for post in posts:
                # 創建一個只包含所需欄位的新字典
                cleaned_post = {
                    "id": post.get("id", ""),
                    "username": post.get("username", ""), # 對應 Google Sheets 的 User_id
                    "text": (post.get("text", "") or "").replace("\n", " "), # 處理文字內容，替換換行符
                    "like_count": post.get("like_count", 0),
                    "reply_count": post.get("reply_count", 0),
                    "timestamp":post.get("timestamp", 0),
                }
                cleaned_posts.append(cleaned_post)
            out = {"posts": cleaned_posts}
            return json.dumps(out, ensure_ascii=False, indent=1)
if __name__ == "__main__":
    with open('config/threadsUser.json','r',encoding='utf-8') as f:
        cfg = json.load(f)
    threads=Threads_scraper(username=["huang.weizhu"])
    threads.filter_setting(gclike=1)
    posts=(asyncio.run(
            threads.Top_crawl(batch=10)
        ))
    #threads.printPost(posts)
    p=json.loads(threads.getJosn(posts))
    print(p)