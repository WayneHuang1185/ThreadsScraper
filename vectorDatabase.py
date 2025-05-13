from pinecone import Pinecone, ServerlessSpec
import json
from sentence_transformers import SentenceTransformer
from Threads import Threads_scraper 
import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional,Dict
class vectorDatabase:
    def __init__(self, index_name:str="threads"):
        with open('config/pinecone.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.model=SentenceTransformer('all-mpnet-base-v2')
        self.pc=Pinecone(api_key=cfg["API_KEY"])
        self.index=self._create_index(index_name=index_name, dimension=self.model.get_sentence_embedding_dimension())
        self.filter=FilterBuilder()
    def _create_index(self, index_name:str, dimension:int=768):
        # Create a new index with the specified name and dimension
        if not self.pc.has_index(index_name):
            self.pc.create_index(
                name=index_name,
                dimension=dimension, 
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                ),
                pod_type="p1"
            )
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1) 
        return self.pc.Index(index_name)
    def embed(self,docs: list[str]) -> list[list[float]]:
        embading=self.model.encode(docs,show_progress_bar=False,convert_to_numpy=True)
        return embading.tolist()
    def store_embeddings_with_tag(self,posts:List[Dict]):
        vectors = []
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
        self.index.upsert(vectors=vectors, namespace="threads")
    def set_filter(self, styles: List[str] = None, username:str=None,min_likes: int = 100, within_days: int = 30):
        self.filter.by_tags(styles).min_likes(min_likes).within_days(within_days).username(username)
    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        filter = self.filter.build()
        emdQuery = self.embed([query])[0]
        response = self.index.query(
            vector=emdQuery,
            top_k=top_k,
            include_metadata=True,
            namespace="threads",
            filter=filter
        )
        return response["matches"]
class FilterBuilder:
    def __init__(self,tags: Optional[List[str]]=None,username:Optional[str]=None,min_likes: int = 100, within_days: int = 30):
        self._tag_clause=tags
        self._min_likes_clause=min_likes
        self._date_clause=within_days
        self._username_clause=username
    def by_tags(self, tags: List[str]) -> "FilterBuilder":
        if tags:
            self._tag_clause={ "tag": { "$in": tags }}
        return self
    def min_likes(self, n: int) -> "FilterBuilder":
        if n is not None:
            self._min_likes_clause={ "like_count": { "$gte": n } }
        return self

    def within_days(self, days: int) -> "FilterBuilder":
        if days is not None:
            cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
            self._date_clause={ "created_at": { "$gte": cutoff } }
        return self
    def username(self, username: str) -> "FilterBuilder":
        if username is not None:
            self._username_clause={ "username": { "$eq": username } }
        return self
    def build(self) -> Optional[dict]:
        clauses = [self._tag_clause, self._min_likes_clause, self._date_clause, self._username_clause]
        if None in clauses:
            clauses.remove(None)
        if not clauses:
            return None
        if len(clauses) == 1:
            return self.clauses[0]
        return { "$and": clauses}
           
        


if __name__ == "__main__":
    threads=Threads_scraper(username=["huang.weizhu"])
    threads.filter_setting(gclike=1)
    posts=(asyncio.run(
            threads.Top_crawl(batch=5)
        ))
    #threads.printPost(posts)
    p=json.loads(threads.getJosn(posts))
    posts_text = [post['text'] for post in p['posts']]