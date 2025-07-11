from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import logging
import os
from fastapi.middleware.cors import CORSMiddleware
from workflow_module import Workflow
from threadsPost import ThreadsAPI
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
import pytz, uuid
import json
from schedule import FireStore
# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Threads 內容產生 API",
    description="用於生成 Threads 風格文章的 API",
    version="1.0.0"
)
# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境中應設置特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局工作流實例

# tz = pytz.timezone("Asia/Taipei")
# scheduler = AsyncIOScheduler(timezone=tz)

# 定義數據模型
class GenerateRequest(BaseModel):
    userquery: str = Field(..., description="文章主題")
    style: str = Field(..., description="文章風格")
    size: int = Field(..., description="文章長度")
    tag: Optional[str] = Field("", description="標籤")
    withindays: Optional[int] = Field(30, description="天數限制")
    gclikes: Optional[int] = Field(1000, description="最小讚數")
    recommendation: Optional[int] = Field(3, description="參考數量")
    specific_user: Optional[str] = Field(None, description="特定用戶名稱")
    @field_validator('style')
    def validate_style(cls, v, values, **kwargs):
        if v not in workflow.config.valid_style:
            raise ValueError(f"Invalid style. Must be one of: {', '.join(workflow.config.valid_style)}")
        return v
    
    @field_validator('size')
    def validate_size(cls, v, values, **kwargs):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("Size must be a positive integer")
        return v


class ToneRequest(BaseModel):
    tone: str = Field(..., description="語氣風格")
class rankWeightRequest(BaseModel):
    weight_type: str = Field(..., description="權重種類")
    @field_validator('weight_type')
    def validate_weight(cls, v, values, **kwargs):
        if v not in workflow.config.rankweight.keys():
            raise ValueError(f"Invalid weight type. Must be one of: {', '.join(workflow.config.rankweight.keys())}")
        return v

class SetWeight(BaseModel):
    relevance:float
    traffic:float
    recency:float
class ApiResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    post: Optional[Any] = None
    styles: Optional[List[str]] = None
    tones  : Optional[List[dict]] = None
    weight : Optional[dict[str,float]] = None

# 依賴函數

def get_workflow():
    return workflow
def get_threads():
    return threads
@app.on_event("startup")
async def start_scheduler():
    global workflow
    global threads
    global firestore
    workflow = Workflow()
    threads=ThreadsAPI()
    firestore=FireStore()
@app.on_event("shutdown")
async def shutdown_scheduler():
    firestore.schedule.shutdown(wait=False)
    firestore.watch.unsubscribe()
@app.get("/", response_model=ApiResponse)
async def root():
    return ApiResponse(success=True, message="歡迎使用 Threads 內容產生 API")
@app.get("/get_weight",response_model=ApiResponse)
async def get_weight():
    try:
        with open('rankweight.json','r',encoding='utf-8') as f:
            data=json.load(f)
        return ApiResponse(success=True,weight=data)
    except Exception as e:
        logger.error(f"取得權重時發生錯誤: {str(e)}")
        return ApiResponse(success=False, error=str(e))
@app.post("/generate", response_model=ApiResponse)
async def generate(request: GenerateRequest, workflow: Workflow = Depends(get_workflow)):
    """生成貼文
    
    根據提供的參數生成符合特定風格的貼文
    """
    try:
        logger.info(f"產生風格為 '{request.style}' 的貼文: {request.userquery}") 
        logger.debug(f"參數: size={request.size}, tag={request.tag}, withindays={request.withindays}, gclikes={request.gclikes}, recommendation={request.recommendation}")
        post = workflow.generate_post(
            userquery=request.userquery,
            style=request.style,
            size=request.size,
            tag=request.tag,
            withindays=request.withindays,
            gclike=request.gclikes,
            recommendation=request.recommendation
        )
        if post == "EOF":
            return ApiResponse(success=False, error="Failed to generate post after maximum attempts")
        
        logger.info("貼文生成成功")
        return ApiResponse(success=True, post=post)

    except Exception as e:
        logger.error(f"生成貼文時發生錯誤: {str(e)}")
        return ApiResponse(success=False, error=str(e))
@app.post("/generate_specific_user", response_model=ApiResponse)
async def generate_specific_user(request: GenerateRequest, workflow: Workflow = Depends(get_workflow)):
    """生成特定用戶的貼文"""
    try:
        if not request.specific_user:
            return ApiResponse(success=False, error="specific_user cannot be null")
        logger.info(f"產生特定用戶 '{request.specific_user}' 的貼文") 
        post = workflow.generate_specific_user_post(
            userquery=request.userquery,
            style=request.style,
            size=request.size,
            tag=request.tag,
            withindays=request.withindays,
            recommendation=request.recommendation,
            specific_user=request.specific_user
        )
        if post == "EOF":
            return ApiResponse(success=False, error="Failed to generate post after maximum attempts")
        logger.info("特定用戶貼文生成成功")
        return ApiResponse(success=True, post=post)
    except Exception as e:
        logger.error(f"生成特定用戶貼文時發生錯誤: {str(e)}")
        return ApiResponse(success=False, error=str(e))
@app.post("/scrape", response_model=ApiResponse)
async def scrape(workflow: Workflow = Depends(get_workflow)):
    """抓取新貼文並存入資料庫"""
    try:
        await workflow.tagging_new_scrape_posts_into_pinecone()
        return ApiResponse(success=True, message="Successfully scraped and processed posts")
    except Exception as e:
        logger.error(f"抓取貼文時發生錯誤: {str(e)}")
        return ApiResponse(success=False, error=str(e))

@app.get("/styles", response_model=ApiResponse)
async def get_styles(workflow: Workflow = Depends(get_workflow)):
    """獲取可用的文章風格"""
    return ApiResponse(success=True, styles=workflow.config.valid_style)

@app.post("/tone", response_model=ApiResponse)
async def change_tone(request: ToneRequest, workflow: Workflow = Depends(get_workflow)):
    """格式
    {"tone": "name"}
    """
    """變更語氣風格"""
    try:
        if workflow.change_tone(request.tone):
            logger.info("語氣已改變")
            return ApiResponse(success=True, message=f"successfully change into {request.tone}'s tone")
        else:
            return ApiResponse(success=False, error=f"{request.tone} does not exist")
    except Exception as e:
        logger.error(f"變更語氣時發生錯誤: {str(e)}")
        return ApiResponse(success=False, error=str(e))

@app.get("/valid_tone",response_model=ApiResponse)
async def get_tone(workflow: Workflow = Depends(get_workflow)):
        tonedata=[]
        for t in workflow.config.valid_tone:
            item={
                'id':t,
                'name':workflow.config.character_name[t],
                'description':workflow.config.character_decription[t]
            }
            tonedata.append(item)
        logger.info("獲取可用的語氣風格資訊")
        return ApiResponse(success=True, tones=tonedata)
@app.post("/change_weight", response_model=ApiResponse)
async def change_weight(request:rankWeightRequest,workflow: Workflow = Depends(get_workflow)):
    """格式
    {"weight_type":'relevance or recency or traffic'}
    """
    """變更權重"""
    if workflow.change_weight(request.weight_type):
        logger.info(f"權重已改變為 {request.weight_type}")
        return ApiResponse(success=True, message=f"successfully change into {request.weight_type}'s weight")
    else:
        return ApiResponse(success=False, error=f"{request.weight_type} does not exist")
@app.post("/generate_specific_user", response_model=ApiResponse)
async def generate_specific_user(request: GenerateRequest, workflow: Workflow = Depends(get_workflow)):
    """生成特定用戶的貼文"""
    try:
        if not request.specific_user:
            return ApiResponse(success=False, error="specific_user cannot be null")
        logger.info(f"產生特定用戶 '{request.specific_user}' 的貼文") 
        post = await workflow.generate_specific_user(
            userquery=request.userquery,
            size=request.size,
            tag=request.tag,
            recommendation=request.recommendation,
            username=request.specific_user
        )
        if post == "EOF":
            return ApiResponse(success=False, error="Failed to generate post after maximum attempts")
        logger.info("特定用戶貼文生成成功")
        return ApiResponse(success=True, post=post)
    except Exception as e:
        logger.error(f"生成特定用戶貼文時發生錯誤: {str(e)}")
        return ApiResponse(success=False, error=str(e))
@app.post("/set_weight",response_model=ApiResponse)
async def SetWeight(request:SetWeight,workflow: Workflow = Depends(get_workflow)):
    try:
        recency = request.recency
        traffic = request.traffic
        relevance = request.relevance
        workflow.set_weight(relevance=relevance, traffic=traffic, recency=recency)
        logger.info(f"已設定新權重: relevance={relevance}, traffic={traffic}, recency={recency}")
        return ApiResponse(success=True, message="Successfully set new weights")
    except Exception as e:
        logger.error(f"設定特定權重時發生錯誤: {str(e)}")
        return ApiResponse(success=False, error=str(e))

# @app.post("/post", response_model=ApiResponse)
# async def post_article(
#     request: Post,
#     threads: ThreadsAPI = Depends(get_threads)
# ):
#     try:
#         # 若未指定時間或時間在過去 -> 立即發布
#         if request.schedule == None:
#             target_time=None
#         else:
#             target_time = request.run_at
#         if not target_time or target_time <= datetime.now(tz):
#             threads.publish_text(request.text)
#             logger.info("文章已立即發布")
#             return ApiResponse(success=True, message="Article posted immediately")

#         # 有指定未來時間 -> 排程
#         job_id = f"post-{uuid.uuid4()}"
#         scheduler.add_job(
#             threads.publish_text,
#             trigger=DateTrigger(run_date=target_time),
#             args=[request.text],
#             id=job_id,
#             misfire_grace_time=300  # 5 分鐘容錯
#         )
#         logger.info(f"文章已排程，Job ID: {job_id}，將於 {target_time.isoformat()} 發布")
#         return ApiResponse(
#             success=True,
#             message=f"Article scheduled at {target_time.isoformat()} (job {job_id})"
#         )
#     except Exception as e:
#         logger.error(f"發布文章發生錯誤: {str(e)}")
#         return ApiResponse(success=False, error=str(e))
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=port, reload=False) 