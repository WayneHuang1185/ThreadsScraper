# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from workflow_module import Workflow
import logging
from typing import Dict, Any, Optional
import os
import config
# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局變量
workflow = Workflow()

def validate_request_data(data: Dict[str, Any]) -> Optional[str]:
    """驗證請求數據"""
    required_fields = ["userquery", "style", "size"]
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"
    
    if not isinstance(data["size"], int) or data["size"] <= 0:
        return "Size must be a positive integer"
    
    if data["style"] not in workflow.config.valid_style:
        return f"Invalid style. Must be one of: {', '.join(workflow.config.valid_style)}"
    return None

@app.route("/generate", methods=["POST"])
def generate():
    """生成貼文
    
    POST /generate
    {
        "userquery": "文章主題",
        "style": "文章風格",
        "size": 文章長度,
        "tag": "標籤",
        "withindays": 天數限制,
        "gclikes": 最小讚數,
        "top_k": 參考數量
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # 驗證請求數據
        error = validate_request_data(data)
        if error:
            return jsonify({
                "success": False,
                "error": error
            }), 400

        # 設置默認值
        withindays = data.get("withindays", 30)
        gclikes = data.get("gclikes", 1000)
        top_k = data.get("top_k", 5)

        # 生成貼文
        post = workflow.generate_post(
            userquery=data["userquery"],
            style=data["style"],
            size=data["size"],
            tag=data.get("tag",""),
            withindays=withindays,
            gclike=gclikes,
            top_k=top_k
        )

        if post == "EOF":
            return jsonify({
                "success": False,
                "error": "Failed to generate post after maximum attempts"
            }), 500

        return jsonify({
            "success": True,
            "post": post
        })

    except Exception as e:
        logger.error(f"Error generating post: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/scrape", methods=["POST"])
def scrape():
    """抓取新貼文並存入資料庫"""
    try:
        workflow.tagging_new_scrape_posts_into_pinecone()
        return jsonify({
            "success": True,
            "message": "Successfully scraped and processed posts"
        })
    except Exception as e:
        logger.error(f"Error scraping posts: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/styles", methods=["GET"])
def get_styles():
    """獲取可用的文章風格"""
    return jsonify({
        "success": True,
        "styles": workflow.config.valid_style
    })
@app.route("/tone",methods=["POST"])
def change_tone():
    """ POST/tone
    {
        "tone":語氣風格,
    } """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        if(workflow.select_character_mode(data["tone"])):
            return jsonify({
                "success": True,
                "message": f"successfully change into {data['tone']}'s tone"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"{data['tone']} does not exist"
            }), 400
    except Exception as e:
        logger.error(f"Error changing tone: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)