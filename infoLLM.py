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
        self.evaluate_prompt=""
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
                4. **產生的內容一定要正確符合邏輯，這非常的重要**
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
    def set_evaluate_prompt(self,userquery,style,response,fewshot):
         self.evaluate_prompt=f"""
                你是一位專業的社群分析師，專門評估貼文是否具備高流量潛力。
                你將會看到一則貼文，包含其主題分類、貼文文字本身、使用者對於文章的訴求。
                工作流程分為構思、評估、擇優和產出
                【評分準則】  
                1. 情緒共鳴（Emotion）：是否能真實勾起讀者情感共鳴？  
                2. 主題貼合度（Relevance）：是否緊扣使用者需求或分類？  
                3. 文字流暢度（Readability）：句子是否通順、易懂，適合 Threads 閱讀？  
                4. 吸睛效果（Engagement Potential）：開頭是否具有吸引力？  
                5. 流量效果(High traffic)是否具有和高流量文章相似的文章結構？
                【工作流程】
                Step 1 ── Generate Thoughts
                 - 為每一評分準則（emotion / relevance / readability / engagement / traffic）
                   產生 3 條不同推理路徑：Thought-A / Thought-B / Thought-C
                 - 每條推理≤40字，列出該準則的優缺點評析
                Step 2 ── Rate Thoughts
                 - 為同一準則的三條 Thought 分別打分 0–1（保留二位小數）
                Step 3 ── Select Best
                 - 每個準則只保留得分最高的 Thought，作為最終依據
                Step 4 ── Compute Scores
                 - 依保留的最佳 Thought，給該準則一個 0–1 分（兩位小數）
                 - 綜合五項準則，算出加權平均（各佔 20 %）→ 最終分數需要正規化到 0–1 範圍
                 - 準備 comments：取最佳 Thought 並縮成 ≤20 字
                Step 5 ── Discard Scratchpad
                 - 禁止在最終回覆中顯示任何 “Thought” 或打分過程  
                【使用者輸入】
                使用者需求：{userquery}
                使用者風格：{style}
                高流量文章範例：{fewshot}
                欲評估貼文：{response}  
                【輸出格式】
                ```json{{
                "comments": "<評估意見，20字以內>",
                "score": <評估分數，範圍 0~1，兩位小數>
                }}
                """
         
