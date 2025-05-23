from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import logging
import os
from fastapi.middleware.cors import CORSMiddleware
from workflow_module import Workflow
from threadsPost import ThreadsAPI
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
workflow = Workflow()
threads=ThreadsAPI()
# 定義數據模型
class GenerateRequest(BaseModel):
    userquery: str = Field(..., description="文章主題")
    style: str = Field(..., description="文章風格")
    size: int = Field(..., description="文章長度")
    tag: Optional[str] = Field("", description="標籤")
    withindays: Optional[int] = Field(30, description="天數限制")
    gclikes: Optional[int] = Field(1000, description="最小讚數")
    top_k: Optional[int] = Field(5, description="參考數量")
    
    @validator('style')
    def validate_style(cls, v, values, **kwargs):
        if v not in workflow.config.valid_style:
            raise ValueError(f"Invalid style. Must be one of: {', '.join(workflow.config.valid_style)}")
        return v
    
    @validator('size')
    def validate_size(cls, v, values, **kwargs):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("Size must be a positive integer")
        return v

class ToneRequest(BaseModel):
    tone: str = Field(..., description="語氣風格")
class Post(BaseModel):
    text: str = Field(..., description="文章內容")
class ApiResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    post: Optional[Any] = None
    styles: Optional[List[str]] = None
    tones  : Optional[List[str]] = None

# 依賴函數
def get_workflow():
    return workflow
def get_threads():
    return threads
@app.get("/", response_model=ApiResponse)
async def root():
    return ApiResponse(success=True, message="歡迎使用 Threads 內容產生 API")

@app.post("/generate", response_model=ApiResponse)
async def generate(request: GenerateRequest, workflow: Workflow = Depends(get_workflow)):
    """生成貼文
    
    根據提供的參數生成符合特定風格的貼文
    """
    try:
        logger.info(f"產生風格為 '{request.style}' 的貼文: {request.userquery}") 
        post = workflow.generate_post(
            userquery=request.userquery,
            style=request.style,
            size=request.size,
            tag=request.tag,
            withindays=request.withindays,
            gclike=request.gclikes,
            top_k=request.top_k
        )

        if post == "EOF":
            return ApiResponse(success=False, error="Failed to generate post after maximum attempts")
        
        logger.info("貼文生成成功")
        return ApiResponse(success=True, post=post)

    except Exception as e:
        logger.error(f"生成貼文時發生錯誤: {str(e)}")
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
        return ApiResponse(success=True, tones=workflow.config.valid_tone)

# @app.post("/post",response_model=ApiResponse)
# async def post_article(request:Post, threads: ThreadsAPI = Depends(get_threads)):
#     try:
#         if request.text:
#             threads.publish_text(request.text)
#             logger.info("文章已發布")
#             return ApiResponse(success=True,message="successfully post article")
#         else:
#             return ApiResponse(success=False,message="content is null")
#     except Exception as e:
#         logger.error(f"發布文章發生錯誤: {str(e)}")
#         return ApiResponse(success=False, error=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=port, reload=False) 