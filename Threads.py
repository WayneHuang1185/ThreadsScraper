"""Threads Scraper (with total post count)

支援：
* 單一帳號或多帳號批次
* --json 輸出原始 JSON
* 額外顯示每個帳號的 **總貼文數**（若能從 JSON 取得）
"""

import asyncio
import json
from typing import List, Dict, Optional
import jmespath
from nested_lookup import nested_lookup
from parsel import Selector
from playwright.async_api import async_playwright
import logging
from datetime import datetime

# 設定 logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Threads_scraper:
    def __init__(self, search_choice: str = "like_count", acc: bool = False, username: Optional[List[str]] = None):
        try:
            with open('existID.json', 'r', encoding='utf-8') as f:
                self.seen = set(json.load(f))
                logger.info(f"已載入 existID.json，共有 {len(self.seen)} 篇已過濾")
        except (FileNotFoundError, json.JSONDecodeError):
            self.seen = set()
            logger.warning("找不到 existID.json 或格式錯誤，初始化為空集合")

        self.url = "https://www.threads.net/@"
        self.QUERY_SELECTOR = "script[type='application/json'][data-sjs]"
        self.search_choice = search_choice
        self.acc = acc
        self.gclike = 0
        self.gcreply = 0
        self.lttext = 0
        self.image_retrieve = False
        self.username = username

    def filter_setting(self, gclike: int = 0, gcreply: int = 0, lttext: int = 0, image_retrieve: bool = False):
        self.gclike = max(0, gclike)
        self.gcreply = max(0, gcreply)
        self.lttext = max(0, lttext)
        self.image_retrieve = image_retrieve
        logger.info(f"設定過濾條件：like ≥ {self.gclike}, reply ≥ {self.gcreply}, text ≤ {self.lttext}, 圖片：{self.image_retrieve}")

    def _parse_post(self, item: Dict) -> Dict:
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

    def _sort_posts(self, posts: List[Dict], sort_key: str, ascending: bool = False) -> List[Dict]:
        return sorted(posts, key=lambda p: p.get(sort_key) or 0, reverse=not ascending)

    async def Top_crawl(self, batch: int = 3) -> List[Dict]:
        logger.info(f"開始爬蟲，共 {len(self.username)} 個帳號，每個取 {batch} 篇")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            usertmp = self.username.copy()
            posts: List[Dict] = []

            for idx, username in enumerate(self.username):
                url = self.url + username
                logger.info(f"[{idx+1}/{len(self.username)}] 正在抓取：{url}")
                try:
                    await page.goto(url, timeout=60000)
                    await page.wait_for_selector(self.QUERY_SELECTOR, state="attached", timeout=60000)
                except Exception as e:
                    logger.error(f"無法載入 {url}：{e}")
                    continue

                selector = Selector(await page.content())
                scripts = selector.css(f"{self.QUERY_SELECTOR}::text").getall()
                done = False
                pre = len(posts)

                for payload in scripts:
                    if '"thread_items"' not in payload:
                        continue

                    data = json.loads(payload)
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
                            if parsed["id"] not in self.seen:
                                posts.append(parsed)
                                if len(posts) >= (idx + 1) * batch:
                                    break
                        if len(posts) >= (idx + 1) * batch:
                            done = True
                            break
                    if done:
                        break

                if len(posts) == pre:
                    usertmp.remove(username)
                    logger.warning(f"帳號 {username} 沒有抓到符合條件的貼文")

            self.username = usertmp
            posts = self._sort_posts(posts, self.search_choice, self.acc)
            logger.info(f"共抓到 {len(posts)} 篇貼文")
            return posts

    def printPost(self, posts):
        for idx, post in enumerate(posts, 1):
            excerpt = (post["text"] or "").replace("\n", " ")[:150]
            url = f"https://www.threads.net/@{post['username']}/post/{post['code']}"
            logger.info(f"{idx}. {excerpt}…\n   👍 {post['like_count']}   💬 {post['reply_count']}   {url}")

    def printJosn(self, posts):
        out = {"post": posts}
        logger.info(json.dumps(out, ensure_ascii=False, indent=1))

    def getJosn(self, posts):
        cleaned_posts = []
        for post in posts:
            cleaned_post = {
                "id": post.get("id", ""),
                "username": post.get("username", ""),
                "text": (post.get("text", "") or "").replace("\n", " "),
                "like_count": post.get("like_count", 0),
                "reply_count": post.get("reply_count", 0),
                "timestamp": post.get("timestamp", 0),
            }
            cleaned_posts.append(cleaned_post)
        out = {"posts": cleaned_posts}
        return json.dumps(out, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    logger.info("載入 threadsUser 設定")
    with open('config/threadsUser.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    threads = Threads_scraper(username=["huang.weizhu"])
    threads.filter_setting(gclike=1)
    posts = asyncio.run(threads.Top_crawl(batch=10))
    p = json.loads(threads.getJosn(posts))
    logger.info("完成爬蟲輸出")
    logger.info(json.dumps(p, ensure_ascii=False, indent=1))
