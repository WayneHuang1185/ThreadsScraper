from pinecone import Pinecone, ServerlessSpec
import json
from sentence_transformers import SentenceTransformer
from Threads import Threads_scraper 
import asyncio
import time
import logging
from config import PINECONE_API_KEY
from datetime import datetime, timedelta, timezone
from typing import List, Optional,Dict

logger = logging.getLogger("pinecone")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s"
))
logger.addHandler(handler)

class vectorDatabase:
    def __init__(self, index_name:str="threads"):
        logger.info("Initializing vectorDatabase with index '%s'", index_name)
        self.model=SentenceTransformer('all-mpnet-base-v2')
        self.pc=Pinecone(PINECONE_API_KEY)
        # print(self.model.get_sentence_embedding_dimension())
        self.index=self._create_index(index_name=index_name, dimension=self.model.get_sentence_embedding_dimension())
        self.filter=FilterBuilder()
    def _create_index(self, index_name:str, dimension:int=768):
        # Create a new index with the specified name and dimension
        logger.debug("Checking if index '%s' exists...", index_name)
        if not self.pc.has_index(index_name):
            self.pc.create_index(
                name=index_name,
                dimension=dimension, 
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                ),
            )
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1) 
            logger.info("Index '%s' is now ready.", index_name)
        else:
            logger.info("Index '%s' already exists.", index_name)
        return self.pc.Index(index_name)
    def embed(self,docs: list[str]) -> list[list[float]]:
        logger.debug("Embedding %d documents...", len(docs))
        embading=self.model.encode(docs,show_progress_bar=False,convert_to_numpy=True)
        return embading.tolist()
    def store_embeddings_with_tag(self,posts:List[Dict]):
        logger.info("Storing %d embeddings to Pinecone...", len(posts))
        vectors = []
        try:
            with open('existID.json', 'r', encoding='utf-8') as f:
                id = set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
                id = set()
        for post in posts:
            post_id = post['id']
            text = post['text']
            username = post['username']
            timpestamp = post['timestamp']
            like_count = post['like_count']
            tag = post['tags']
            vec = self.embed([text])[0]
            vectors.append({
                "id": post_id,
                "values": vec,
                "metadata": {
                    "text": text,
                    "username": username,
                    "created_at":timpestamp,
                    "tag":tag,
                    "like_count":like_count,
                }
            })
            id.add(post_id)
        try:
            resp=self.index.upsert(vectors=vectors, namespace="threads")
            logger.info("Upserted %d vectors (upsert_response: %s)", len(vectors), resp)
            with open('existID.json', 'w', encoding='utf-8') as f:
                json.dump(list(id), f, ensure_ascii=False, indent=1)
            logger.debug("Wrote %d IDs to existID.json", len(posts))
        except Exception as e:
            logger.exception("Failed to upsert vectors or write existID.json")

    def set_filter(self, styles: List[str] = None, username:str=None,min_likes: int = 100, within_days: int = 30):
        self.filter = FilterBuilder()
        logger.debug("Setting filter: styles=%s, username=%s, min_likes=%s, within_days=%s",
                     styles, username, min_likes, within_days)
        self.filter.by_tags(styles).min_likes(min_likes).within_days(within_days).username(username)
    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        logger.info("Querying top %d for '%s'...", top_k, query)
        filter = self.filter.build()
        emdQuery = self.embed([query])[0]
        response = self.index.query(
            vector=emdQuery,
            top_k=top_k,
            include_metadata=True,
            namespace="threads",
            filter=filter
        )
        matches=response["matches"]
        logger.debug("Query returned %d matches", len(matches))
        return matches
class FilterBuilder:
    def __init__(self):
        self.clause=[]
    def by_tags(self, tags: List[str]) -> "FilterBuilder":
        if tags:
            self.clause.append({ "tag": { "$in": tags }})
        return self
    def min_likes(self, n: int) -> "FilterBuilder":
        if n is not None:
            self.clause.append({ "like_count": { "$gte": n } })
        return self

    def within_days(self, days: int) -> "FilterBuilder":
        if days is not None:
            cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
            self.clause.append({ "created_at": { "$gte": cutoff } })
        return self
    def username(self, username: str) -> "FilterBuilder":
        if username is not None:
            self.clause.append({ "username": { "$eq": username } })
        return self
    def build(self) -> Optional[dict]:
        if None in self.clause:
            self.clause.remove(None)
        if len(self.clause)<=0:
            return None
        if len(self.clause) == 1:
            return self.clause[0]
        return { "$and": self.clause}
           
        
if __name__ == "__main__":
    vd=vectorDatabase()
    vd.set_filter(username='huang.weizhu',min_likes=1)
    text=vd.query('好想放假，不想考試',top_k=10)
    print(text)