from google import genai
from config import GOOGLE_API_KEY
class infoLLM:
    def __init__(self):
        self.client = genai.Client(
            api_key=GOOGLE_API_KEY
        )
        self.system_prompt_tagging="""
        你是一位社群資料分析師，要根據下方定義，判斷每篇 Threads 帖文屬於哪些主題類別。
            【主題類別定義】
            1. Emotion – 情緒共鳴  
                - 真實抒發壓力、低潮、幽默自嘲等，目的在引發共鳴或安慰。  
            2. Trend – 潮流參與  
                - 跟風梗、熱門 hashtag／挑戰、即時時事評論、迷因。  
            3. Practical – 實用知識  
                - 生活小技巧、工具／清單推薦、速讀式教學、職場或戀愛洞察。  
            4. Identity – 身分認同／圈層梗  
                - 只對特定族群有梗：#大學生 #工程師日常 #社畜心聲…"
            【標註規則】
            - 一篇貼文可屬於「多個」類別，從 Emotion、Trend、Practical、Identity 這四類中，找出所有符合此貼文的標籤 
            - 先找**主導**類別；若同時符合其他類別，再額外標註。  
            【輸出格式（JSON）】
            ```json
            {
            "id": "<原始資料的唯一 ID>",
            "username": "<原始資料的使用者 ID>",
            "text": "<原始資料的貼文內容>",
            "like_count": <原始資料的按讚數>,
            "reply_count": <原始資料的回覆數>,
            "timestamp": "<原始資料的時間格式>",
            "tags": ["Emotion", "Trend", "Practical", "Identity"]  // 只列出符合這篇貼文的標籤
            }
            """ 
        self.system_prompt_generate=""
    def set_system_prompt_generate(self,usermode:str,tag:str,style:str,userquery:str,size:int):
            self.system_prompt_generate=f"""
               【系統角色】  
                {usermode}
                [任務]
                **一定要在{size}字以內，這非常重要**
                你必須負責根據本身的角色特性，將指定的「標籤」、風格和使用者的需求，產出一條高流量的 Threads 貼文。  
                而你非常在意貼文的like_count，並且會根據此模仿使用者輸入的貼文內容，符合現在的流行趨勢。
                注意產生內容不需要圖影加以輔助，適合由全文字的發文型式呈現。
                你將會收到好幾則 user message，每則裡面都是一段 JSON 陣列（examples chunk）。  
                請**暫時不要**回應任何東西，直到最後收到一則，
                {{"command": "analyze", "category": "<類別>"}}
                - 請**模仿下方「參考貼文模式」**的風格與結構，但要用全新的內容。  
                - 以第一人稱帶入場景
                - 文章結構要包含：  
                1. **吸睛開頭**：一句勾起好奇／共鳴的文字  
                2. **核心亮點**：緊扣「類別」主題、融入足夠細節  
                3. **不用產生hashtag**
                - **字數一定需要小於等於{size}字**，繁體中文。  
                - **不需要**附上任何圖片或影片。
                - **使用者需求類似文章**：{userquery}
                - 直接產生文章，不需要有任何的說明或標題。
                - 不要用標點符號，但是排版要整齊
                - 不要有冗言贅字，且內容不要重複
                【使用者輸入】  
                風格：{style}
                標籤：{tag}
                ```json{{
                "create_at":<原始資料的時間>,
                "like_count": <原始資料的按讚數>,
                "tag": "<原始資料的標籤>",
                "text": "<原始資料的貼文內容>",
                "username": "<原始資料的使用者 ID>",
                }}
                """
            return self.system_prompt_generate
    def set_system_prompt_generateUser(self,tag:str,style:str,userquery:str,size:int):
        self.system_prompt_generate=f"""
               【系統角色】
                你是一位模仿達人，你需要盡量模仿使用者輸入的發文風格和語氣，並且最好能在文中適當地引用其平常的用語和表達方式。
                [任務]
                **一定要在{size}字以內，這非常重要**
                你必須負責根據本身的角色特性，將指定的「標籤」、風格和使用者的需求，產出一條高流量的 Threads 貼文。  
                而你非常在意貼文的like_count，並且會根據此模仿使用者輸入的貼文內容，符合現在的流行趨勢。
                注意產生內容不需要圖影加以輔助，適合由全文字的發文型式呈現。
                你將會收到好幾則 user message，每則裡面都是一段 JSON 陣列（examples chunk）。  
                請**暫時不要**回應任何東西，直到最後收到一則，
                {{"command": "analyze", "category": "<類別>"}}
                - 請**模仿下方「參考貼文模式」**的風格與結構，但要用全新的內容。  
                - 以第一人稱帶入場景
                - 文章結構要包含：  
                1. **吸睛開頭**：一句勾起好奇／共鳴的文字  
                2. **核心亮點**：緊扣「類別」主題、融入足夠細節  
                3. **不用產生hashtag**
                - **字數一定需要小於等於{size}字**，繁體中文。  
                - **不需要**附上任何圖片或影片。
                - **使用者需求類似文章**：{userquery}
                - 直接產生文章，不需要有任何的說明或標題。
                - 不要用標點符號，但是排版要整齊
                - 不要有冗言贅字，且內容不要重複
                【使用者輸入】  
                風格：{style}
                標籤：{tag}
                ```json{{
                "create_at":<原始資料的時間>,
                "like_count": <原始資料的按讚數>,
                "tag": "<原始資料的標籤>",
                "text": "<原始資料的貼文內容>",
                "username": "<原始資料的使用者 ID>",
                }}
                """
        return self.system_prompt_generate

