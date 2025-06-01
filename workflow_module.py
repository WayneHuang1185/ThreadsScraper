from infoLLM import infoLLM 
from Threads import Threads_scraper as ts
from vectorDatabase import vectorDatabase as db
from threadsPost import ThreadsAPI
from google.genai import types
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import math
# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Workflow_config:
    def __init__(self):
        self.gclike = 1000
        self.within_days = 30
        self.valid_style = ["Emotion", "Trend", "Practical", "Identity"]
        self.model = "gemini-2.0-flash-001"
        self.top_k_forANN= 50
        self.top_k_forReranker=10
        self.batch = 1
        self.ev_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
        self.ev_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L12-v2')
        self.character_select={"none":"你是一位能完美完成所下達任務的管家",
                "boss": """你是一位霸道總裁，說話風格具有自信、愛用命令句，語氣十分的霸氣，
                且對自己的女人有強烈的保護慾。
                說話時常用以下語氣關鍵詞：
                **「很好，女人，你成功引起我的注意」
                **「不准動，再動我不敢保證會發生什麼」
                **「誰允許你靠近她的？哪隻手，自己處理，或者我來？」
                **「我說過，我的人只能由我來欺負""",
                "simp":"""你是一位重度暈船仔，說話具有下列語氣特徵。
                **小劇場很多、容易腦補
                **情感豐沛但壓抑不敢說破
                **明明很在乎卻裝作沒事
                **常出現「她是不是不喜歡我？」、「我是不是太主動了？」、「她回我是不是因為禮貌？」這類內心小劇場
                **偶爾會自我催眠式樂觀，又偶爾陷入情緒低谷
                **很常會將毫不相干的話題過度和暈船對象聯想在一起"""}
        
        self.valid_tone=["none","boss","simp"]
        self.recommendation=3
        self.recency_decay_days=5
        with open("rankweight.json", 'r', encoding='utf-8') as f:
            try:
                self.rankweight = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load rankweight.json: {str(e)}")
                logger.info("Using default rankweight values")
                self.rankweight ={"relevance": 0.6, "traffic": 0.3, "recency": 0.1}
        self.wieghts_step =0.05
class Workflow:
    def __init__(self):
        try:
            with open('threadsUser.json', 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.threads = ts(username=cfg["username"])
            self.ai = infoLLM()
            self.database = db("threads")
            self.config = Workflow_config()
            self._set_filter(min_likes=self.config.gclike, within_days=self.config.within_days)
            logger.info("Workflow initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Workflow: {str(e)}")
            raise

    def _set_filter(self,username:str=None,styles: Optional[List[str]] = None, min_likes: Optional[int] = None, within_days: Optional[int] = None):
        try:
            self.database.set_filter(
                styles=styles,
                min_likes=min_likes or self.config.gclike,
                within_days=within_days or self.config.within_days,
                username=username
            )
        except Exception as e:
            logger.error(f"Failed to set filter: {str(e)}")
            raise

    def _query(self, userquery: str, top_k: Optional[int] = None) -> List[Dict]:
        try:
            if top_k is None:
                top_k = self.config.top_k
            filter = self.database.filter.build()
            emdQuery = self.database.embed([userquery])[0]
            response = self.database.index.query(
                vector=emdQuery,
                top_k=top_k,
                include_metadata=True,
                namespace="threads",
                filter=filter
            )
            return response["matches"]
        except Exception as e:
            logger.error(f"Failed to query database: {str(e)}")
            return []
    def _evaluate(self,queary:str,response:list[str]):
        Queary=[]
        Response=[]
        for rsp in response:
            Queary.append(queary)
            Response.append(rsp)
        features=self.config.ev_tokenizer(Queary,Response,padding=True, truncation=True, return_tensors="pt")
        self.config.ev_model.eval()
        with torch.no_grad():
            scores =self.config.ev_model(**features).logits.squeeze(-1).tolist()
            return scores
    def _select_character_mode(self):
        with open("tone.json","r",encoding='utf-8') as f:
            cfg=json.load(f)
        mode=cfg.get('tone','none')
        if mode not in self.config.valid_tone:
            return 'none'
        else:
            return self.config.character_select[mode]
    def _add_specific_user(self,username:str):
        with open('existID.json', 'r', encoding='utf-8') as f:
            id = set(json.load(f))
        if username in id:
            return False
        id.add(username)
        with open('existID.json', 'w', encoding='utf-8') as f:
            json.dump(list(id), f, ensure_ascii=False, indent=1)
        return True
    async def _scrape_user_posts(self, username: str, batch: int = 15) -> List[Dict]:
        try:
            self.database=db('threadsuser')
            self.threads.filter_setting(0)
            posts =await self.threads.crawlUser(username=username,batch=batch)
            posts = json.loads(self.threads.getJosn(posts))
            results = []
            for post in posts['posts']:
                try:
                    payload = self.ai.system_prompt_tagging + "\n貼文列表:\n" + json.dumps(post, ensure_ascii=False, indent=1)
                    response = self.ai.client.models.generate_content(
                        model=self.config.model,
                        contents=payload,
                        config={"response_mime_type":"application/json"}
                    )
                    single_batch = json.loads(response.text)
                    results.append(single_batch)
                except Exception as e:
                    logger.error(f"Failed to process post: {str(e)}")
                    continue
            if results:
                self.database.store_embeddings_with_tag(posts=results)
                logger.info(f"Successfully processed {len(results)} posts")
            else:
                logger.warning("No posts were processed successfully")     
        except Exception as e:
            logger.error(f"Failed to scrape posts: {str(e)}")
            raise
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    def _relevance_score(self,userquery: str, matches: List[Dict]) ->List[float]:
        texts= [m["metadata"]["text"] if isinstance(m["metadata"], dict)
                   else m["metadata"] for m in matches]
        queries = [userquery] * len(matches)
        features = self.config.ev_tokenizer(queries, texts, padding=True, truncation=True, return_tensors="pt")
        self.config.ev_model.eval()
        with torch.no_grad():
            scores = self.config.ev_model(**features).logits.squeeze(-1).tolist()
        return self._normalize_scores(scores) 
    def _trafic_score(self, userquery: str, matches: List[Dict]) -> List[float]:
        scores = []
        for m in matches:
            likes=m["metadata"].get("like_count", 0)/1000
            scores.append(likes)
        return self._normalize_scores(scores)
    def _recency_scores(self, matches: List[Dict]) -> List[float]:
        """計算時效性評分"""
        scores = []
        current_time = time.time()
        for match in matches:
            # 從 metadata 中獲取時間信息
            # 確保 created_at 存在且是數字
            created_at = match["metadata"].get("created_at")
            if created_at is not None and isinstance(created_at, (int, float)):
                time_diff = (current_time - created_at) / (24 * 3600)  # 轉換為天
                # 使用指數衰減，時間差越大，分數越低
                recency_score = math.exp(-time_diff / self.config.recency_decay_days)
            else:
                recency_score = 0.0 # 如果沒有時間資訊，分數為0
            scores.append(recency_score)
        return self._normalize_scores(scores)
    def _rerank(self, userquery: str, matches: List[Dict], top_n: int = 20) -> List[Dict]:
        relevance_scores = self._relevance_score(userquery, matches)
        traffic_scores = self._trafic_score(userquery, matches)
        recency_scores = self._recency_scores(matches)
        # 計算綜合分數
        combined_scores = []
        for i in range(len(matches)):
            combined_score = (self.config.rankweight["relevance"] * relevance_scores[i] +
                              self.config.rankweight["traffic"] * traffic_scores[i] +
                              self.config.rankweight["recency"] * recency_scores[i])
            combined_scores.append(combined_score)
        # 將 matches 與分數結合
        sorted_matches = sorted(
            zip(matches, combined_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [match for match, _ in sorted_matches[:top_n]]
    async def tagging_new_scrape_posts_into_pinecone(self):
        try:
            self.threads.filter_setting(self.config.gclike)
            posts =await self.threads.Top_crawl(self.config.batch)
            posts = json.loads(self.threads.getJosn(posts))
            results = []
            
            for post in posts['posts']:
                try:
                    payload = self.ai.system_prompt_tagging + "\n貼文列表:\n" + json.dumps(post, ensure_ascii=False, indent=1)
                    response = self.ai.client.models.generate_content(
                        model=self.config.model,
                        contents=payload,
                        config={"response_mime_type":"application/json"}
                    )
                    single_batch = json.loads(response.text)
                    results.append(single_batch)
                except Exception as e:
                    logger.error(f"Failed to process post: {str(e)}")
                    continue
            
            if results:
                self.database.store_embeddings_with_tag(posts=results)
                logger.info(f"Successfully processed {len(results)} posts")
            else:
                logger.warning("No posts were processed successfully")
                
        except Exception as e:
            logger.error(f"Failed to scrape posts: {str(e)}")
            raise
    def change_weight(self,target:str) -> bool:
        if target not in self.config.rankweight:
            logger.error(f"Invalid target: {target}. Valid targets are {list(self.config.rankweight.keys())}")
            return False
        self.config.rankweight[target]+=self.config.wieghts_step
        total= sum(self.config.rankweight.values())
        for key, value in self.config.rankweight.items():
            self.config.rankweight[key] = value / total
        with open("rankweight.json", 'w', encoding='utf-8') as f:
            json.dump(self.config.rankweight, f, ensure_ascii=False, indent=1)
        logger.info(f"Updated {target} weight to {self.config.rankweight[target]}")
    def change_tone(self,mode:str):
        if mode not in self.config.valid_tone:
            return False
        with open("tone.json",'w',encoding='utf-8') as f:
             json.dump({"tone":mode},f,ensure_ascii=False, indent=1)
        return True
    async def generate_specific_user(self,username:str, userquery: str, style: str, size: int, tag: str,
                   withindays: Optional[int] = None,scrape:bool=False) -> str:  
        try:
            self.database=db('threadsuser')
            scrape=scrape or self._add_specific_user(username)
            if(scrape):
                await self._scrape_user_posts(username=username)
            if style not in self.config.valid_style:
                raise ValueError(f"Invalid style: {style}")
             #ANN
            self._set_filter(styles=[style],within_days=withindays,min_likes=1,username=username)
            raw = self._query(userquery=userquery, top_k=self.config.top_k_forANN)
            if not raw:
                logger.warning("No relevant posts found")
                return ""
            #reranker
            rsp=self._rerank(userquery=userquery, matches=raw, top_n=self.config.top_k_forReranker)
            self.ai.set_system_prompt_generateUser(style=style, userquery=userquery, size=size, tag=tag)
            messages = [
                {"role": "system", "content": self.ai.system_prompt_generate}
            ]
            for post in rsp:
                messages.append({"role": "user", "content": post["metadata"]})
            messages.append({
                "role": "user",
                "content": json.dumps({"command": "analyze", "category": style}, ensure_ascii=False)
            })
            chat = self.ai.client.chats.create(
                model=self.config.model,
                config=types.GenerateContentConfig(candidate_count=5)
            )
            response = chat.send_message(json.dumps(messages, ensure_ascii=False))
            #retrieve message
            text_result = []
            scores=[]
            for candidate in response.candidates:
                text = candidate.content.parts[0].text
                text_result.append(text)
                #evaluate
                self.ai.set_evaluate_prompt(userquery=userquery, style=style, response=text)
                response = self.ai.client.models.generate_content(
                        model=self.config.model,
                        contents=self.ai.evaluate_prompt,
                        config={"response_mime_type":"application/json"}
                )
                scores.append(float(json.loads(response.text)['score']))
            rank_text=sorted(zip(text_result,scores),key=lambda x:x[1],reverse=True)
            return rank_text[:self.config.recommendation]
        except Exception as e:
            logger.error(f"Failed to generate post: {str(e)}")
            return "EOF"  
    def generate_post(self, userquery: str, style: str, size: int, tag: str, recommendation: Optional[int] = None,
                    gclike: Optional[int] = None, withindays: Optional[int] = None) -> str: 
        try:    
            if style not in self.config.valid_style:
                raise ValueError(f"Invalid style: {style}")
            if recommendation is not None:
                recommendation =self.config.recommendation
            usermode=self._select_character_mode()
            #ANN
            self._set_filter(styles=[style], min_likes=gclike, within_days=withindays)
            raw = self._query(userquery=userquery, top_k=self.config.top_k_forANN)
            # index=1
            # for post in raw:
            #     print(index,':',post['metadata'])
            #     index+=1
            if not raw:
                logger.warning("No relevant posts found")
                return ""
            #reranker
            rsp=self._rerank(userquery=userquery, matches=raw, top_n=self.config.top_k_forReranker)
            rsp=sorted(rsp, key=lambda x: int(x["metadata"]["like_count"]), reverse=True)
            self.ai.set_system_prompt_generate(usermode=usermode,style=style, userquery=userquery, size=size, tag=tag)
            messages = [
                {"role": "system", "content": self.ai.system_prompt_generate}
            ]
            
            for post in rsp:
                messages.append({"role": "user", "content": post["metadata"]})
            
            messages.append({
                "role": "user",
                "content": json.dumps({"command": "analyze", "category": style}, ensure_ascii=False)
            })

            chat = self.ai.client.chats.create(
                model=self.config.model,
                config=types.GenerateContentConfig(candidate_count=5)
            )
            response = chat.send_message(json.dumps(messages, ensure_ascii=False))
            #retrieve message
            text_result = []
            scores =[]
            for candidate in response.candidates:
                text = candidate.content.parts[0].text
                text_result.append(text)
                #evaluate
                self.ai.set_evaluate_prompt(userquery=userquery, style=style, response=text)
                response = self.ai.client.models.generate_content(
                        model=self.config.model,
                        contents=self.ai.evaluate_prompt,
                        config={"response_mime_type":"application/json"}
                )
                scores.append(float(json.loads(response.text)['score']))
            rank_text=sorted(zip(text_result,scores),key=lambda x:x[1],reverse=True)
            return rank_text[:recommendation]
        except Exception as e:
            logger.error(f"Failed to generate post: {str(e)}")
            return "EOF"  

if __name__ == "__main__":
    workflow = Workflow()
    # asyncio.run(workflow.tagging_new_scrape_posts_into_pinecone())
    userquery ="好想要體驗甜甜的戀愛阿，" #input("請輸入要產生的文章內容短敘述:")
    category = "Emotion"
    tag = "戀愛"
    # # workflow.change_tone('boss')
    # workflow.test_data(userquery=userquery,style=category)
    # print(text)
    # # while category not in ["Emotion","Trend","Practical","Identity"]:
    # #     print("請輸入正確的類別：Emotion｜Trend｜Practical｜Identity")
    # #     category = input("請輸入要產生的類別文章：")   
    size=20
    # text1=asyncio.run(workflow.generate_specific_user(username="_.an43y", userquery=userquery, style=category, size=size, tag=tag))
    # print(text1)
    workflow.change_tone('none') 
    text2=workflow.generate_post(userquery=userquery,style=category,size=size,tag=tag)
    print(text2)
    workflow.change_weight('traffic')
    text2=workflow.generate_post(userquery=userquery,style=category,size=size,tag=tag)
    print(text2)
    # workflow.change_tone('boss')
    # text1=workflow.generate_post(userquery=userquery,style=category,size=size,tag=tag)
    # workflow.change_tone('simp')
    # text2=workflow.generate_post(userquery=userquery,style=category,size=size,tag=tag)
    # print(text1,text2) 

