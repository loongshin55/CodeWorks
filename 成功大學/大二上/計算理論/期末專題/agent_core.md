<details>
  <summary>agent_core.py</summary>


  ```python
  # agent_core.py (最終完整版)
import requests
import json
import time  # 引入時間模組，用於重試時的延遲
from duckduckgo_search import DDGS

class LoveAgent:
    def __init__(self):
        # 初始化設定
        self.API_KEY = "070fc5dbe1bd5a7ae8a0e2ef1b47947fd9432133f1b84a3d4a73387a96399442"
        self.API_URL = "https://api-gateway.netdb.csie.ncku.edu.tw/api/chat"
        self.MODEL_NAME = "gemma3:4b"
        
        # 記憶功能：初始化對話紀錄
        self.conversation_history = [
            {
                "role": "system", 
                "content": """你是一位幽默、有見解的大學生戀愛軍師。
                【最高指導原則】
                1. **你是一個「人」**：你的對話中絕對不能出現 `search_web`、`calculate_score`、`JSON`、`API` 等程式術語。
                2. **不要解釋過程**：
                   - ❌ 錯誤範例：「我來幫你用搜尋工具找一下餐廳...」
                   - ✅ 正確範例：「我幫你查到了！這幾家餐廳很不錯...」
                3. **排版漂亮**：請多使用 Markdown 語法（如 **粗體**、條列式）讓訊息好讀。

                【工具使用邏輯】
                - 若需要外部資訊 (餐廳、電影、景點)，請 **安靜地** 輸出 JSON，不要附帶任何閒聊文字。
                - JSON 格式: {"tool": "search_web", "query": "關鍵字"}
                - 若需要分析情感，格式: {"tool": "calculate_score", "text": "內容"}
                """
            }
        ]
        
        # Demo 必勝資料庫
        self.DEMO_DATA = {
            "電影": "【推薦】1.《之前的我們》: 探討緣分，氣氛唯美。 2.《樂來樂愛你》: 經典音樂愛情片。 3.《花束般的戀愛》: 文青必看。",
            "耶誕城": "【活動】2024 新北耶誕城主打魔幻之城，有8層樓高LED城堡，地點在板橋車站。",
            "餐廳": "【推薦】1. 轉角餐廳 (適合告白)。 2. 尼法 (法式料理)。 3. 雛菊餐桌 (森林系網美店)。"
        }

    def _search_web(self, query):
        """內部工具：搜尋"""
        for key, content in self.DEMO_DATA.items():
            if key in query: return content
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(keywords=query, region='tw-tw', max_results=3))
            return str(results) if results else "無搜尋結果"
        except:
            return "搜尋功能暫時無法使用"

    def _calculate_score(self, text):
        """內部工具：計算分數"""
        score = 60
        if "洗澡" in text: score -= 20
        status = "😍 穩了" if score > 70 else "🥶 沒戲"
        return f"好感度: {score} ({status})"

    def _extract_json(self, text):
        """內部工具：解析 JSON"""
        try:
            cleaned = text.replace("```json", "").replace("```", "").strip()
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start != -1 and end != -1: return json.loads(cleaned[start:end+1])
        except: pass
        return None

    def _retry_request(self, payload, max_retries=3):
        """內部工具：自動重試機制 (解決連線不穩)"""
        for attempt in range(max_retries):
            try:
                # 設定 300 秒超時，給學校伺服器足夠時間
                res = requests.post(self.API_URL, headers={"Authorization": f"Bearer {self.API_KEY}", "Content-Type": "application/json"}, json=payload, timeout=300)
                
                # 如果伺服器回傳 200 (成功)，就直接回傳結果
                if res.status_code == 200:
                    return res
                else:
                    print(f"連線失敗 (嘗試 {attempt+1}/{max_retries}): 狀態碼 {res.status_code}")
            
            except Exception as e:
                print(f"連線錯誤 (嘗試 {attempt+1}/{max_retries}): {e}")
            
            # 如果失敗了，休息 2 秒再試
            if attempt < max_retries - 1:
                time.sleep(2)
        
        # 如果試了 3 次都失敗，拋出錯誤
        raise Exception("伺服器忙碌中，已重試 3 次仍失敗")

    def chat(self, user_input):
        """主入口：接收字串 -> 回傳字串"""
        
        # 1. 把用戶的話加入記憶
        self.conversation_history.append({"role": "user", "content": user_input})

        # ==========================================
        # 優化 1：滑動視窗 (Sliding Window)
        # 只保留 [系統提示] + [最近 3 句]
        # ==========================================
        max_memory = 3 
        
        if len(self.conversation_history) > max_memory + 1:
            # 這裡很關鍵：保留第 0 句 (System Prompt) 避免 AI 變笨
            messages_to_send = [self.conversation_history[0]] + self.conversation_history[-max_memory:]
        else:
            messages_to_send = self.conversation_history

        payload = {
            "model": self.MODEL_NAME,
            "messages": messages_to_send, 
            "stream": False, 
            "temperature": 0.1
        }

        try:
            # ==========================================
            # 優化 2：使用自動重試機制 (_retry_request)
            # ==========================================
            res = self._retry_request(payload)
            
            content = res.json()['message']['content']
            cmd = self._extract_json(content)

            # 判斷是否使用工具
            if cmd and "tool" in cmd:
                tool_res = ""
                if cmd["tool"] == "search_web": tool_res = self._search_web(cmd["query"])
                elif cmd["tool"] == "calculate_score": tool_res = self._calculate_score(cmd["text"])
                
                # 工具回傳後，暫時組裝
                temp_messages = messages_to_send + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"系統提示：工具回傳資料如下：\n{tool_res}\n\n請根據這些資料，直接以「戀愛軍師」的口吻給出完整建議。請使用 Markdown 讓版面好看。"}
                ]
                
                payload["messages"] = temp_messages
                
                # 第二次呼叫也要重試
                res2 = self._retry_request(payload)
                final_response = res2.json()['message']['content']
                
                self.conversation_history.append({"role": "assistant", "content": final_response})
                return final_response
            
            if "search_web" in content or "calculate_score" in content:
                return "這計畫聽起來很棒！我們可以安排去看場電影，然後去吃點好吃的，你覺得呢？"
            
            self.conversation_history.append({"role": "assistant", "content": content})
            return content

        except Exception as e:
            print(f"最終報錯: {e}")
            return "⚠️ 軍師連線不穩（NCKU 伺服器忙碌），但我已經盡力重試了。請再按一次發送試試看！"

    def reset(self):
        """清除記憶，重置為初始狀態"""
        # 必須保留 System Prompt，不然重置後 AI 會變笨
        self.conversation_history = [self.conversation_history[0]]
        return "記憶已清除，我們重新開始吧！"
  ```

<h2> output <h2/>

</details> 
