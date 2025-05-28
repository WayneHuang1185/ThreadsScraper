
import requests,time,json
from datetime import datetime, timedelta,timezone
from config import THREADS_ACCESS_TOKEN, THREADS_ID
class ThreadsAPI:
    def __init__(self):
        self.access_token = THREADS_ACCESS_TOKEN
        self.user_id =THREADS_ID
    def _create_media_container(self,text=None, media_type="TEXT", image_url=None, video_url=None):
        url = f"https://graph.threads.net/v1.0/{self.user_id}/threads"
        params = {
                "access_token": self.access_token,
                "media_type": media_type,
            }
        if text:
            params["text"] = text
        if media_type == "IMAGE" and image_url:
            params["image_url"] = image_url
        if media_type == "VIDEO" and video_url:
            params["video_url"] = video_url
        resp = requests.post(url, params=params)
        resp.raise_for_status()
        creation_id = resp.json().get("id")
        return creation_id
    
    def _publish_media(self, creation_id):
        url = f"https://graph.threads.net/v1.0/{self.user_id}/threads_publish"
        params = {
            "access_token": self.access_token,
            "creation_id": creation_id,
        }
        resp = requests.post(url, params=params)
        resp.raise_for_status()
        post_id = resp.json().get("id")
        return post_id
    def unix(self,ts: datetime) -> int:
        return int(time.mktime(ts.timetuple()))

    def fetch_user_insights(self,metrics:tuple=("likes","views"),
                            days=7,period="day"):
        until = datetime.now(timezone.utc)
        since = until - timedelta(days=days)
        url = f"https://graph.threads.net/v1.0/{self.user_id}/threads_insights"
        params = {
            "metric": ",".join(metrics),
            "period": period,
            "since": self.unix(since),
            "until": self.unix(until),
            "access_token": self.access_token,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    def retrieve_user_insights(self, metrics: tuple = ("likes", "views"),
                               days=7, period="day"):
        insight_data=self.fetch_user_insights(metrics=metrics, days=days, period=period)
        if insight_data is not None:
            insight_data=insight_data.get("data")
        else:
            return None
        result ={}
        for data in insight_data:
            if(data.get("name") == "views"):
                day_views=[]
                for dv in data.get("values"):
                    time= dv.get("end_time")[:10]
                    day_view=dv.get("value")
                    day_views.append({"time":time,"day_view":day_view})
                result["views"]=day_views
            elif(data.get("name") == "likes"):
                result["likes"]=data.get("total_value")
        return result.json()
        
    def publish_text(self, text):
        creation_id = self._create_media_container(text=text, media_type="TEXT")
        post_id = self._publish_media(creation_id)
        print("發文成功！")
        return post_id
if __name__ == "__main__":
    threads = ThreadsAPI()
    # threads.publish_text("這是測試貼文")
    # threads.fetch_user_insights()
    # threads._create_media_container(text="這是測試貼文", media_type="TEXT")
    # threads._publish_media("1234567890")
    print(threads.fetch_user_insights())