簡介

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
</details>

<details>
  <summary>server.py</summary>

  ```python
  # server.py (不用動，確認內容即可)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_core import LoveAgent

app = FastAPI()
agent = LoveAgent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    print(f"收到前端訊息: {req.message}")
    response_text = agent.chat(req.message)
    return {"reply": response_text}

@app.post("/api/reset")
async def reset_endpoint():
    print("收到重置請求...")
    msg = agent.reset()
    return {"reply": msg}
  ```

</details> 

<details>
  <summary>index.html</summary>

  ```html
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>💘 戀愛軍師 AI</title>
    <link rel="stylesheet" href="style.css">
    
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; font-size: 24px;">💘 戀愛軍師 AI</h1>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">你的專屬感情顧問</p>
                </div>
                
                <button onclick="resetChat()" style="background: rgba(255,255,255,0.3); border: 1px solid white; color: white; padding: 6px 15px; border-radius: 20px; cursor: pointer; font-weight: bold; font-size: 14px; transition: 0.2s;">
                    🔄 重新開始
                </button>
            </div>
        </div>

        <div class="action-buttons">
            <button class="action-btn" onclick="setMode('date')">📅 安排約會</button>
            <button class="action-btn" onclick="setMode('analysis')">🧐 對話分析</button>
            <button class="action-btn" onclick="setMode('horoscope')">🔮 戀愛運勢</button>
        </div>

        <div class="chat-box" id="chat-box">
            <div class="message bot-message">你好！我是你的戀愛軍師，請點擊上方按鈕選擇服務，或直接跟我聊天！</div>
        </div>

        <div class="input-area">
            <input type="text" id="user-input" placeholder="輸入你的問題..." />
            <button class="send-btn" onclick="sendMessage()">發送</button>
        </div>
    </div>

    <script src="script.js"></script>
</body>
</html>
  ```

</details> 

<details>
  <summary>style.css</summary>

  ```css
/* frontend/style.css */
body {
    background-color: #fce4ec;
    font-family: "Microsoft JhengHei", sans-serif;
    display: flex;
    justify-content: center;
    height: 100vh;
    margin: 0;
}

.chat-container {
    width: 400px;
    height: 600px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    margin-top: 50px;
}

.header {
    background: #ff4081;
    color: white;
    padding: 20px;
    text-align: center;
}

.chat-box {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    background: #fafafa;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.message {
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 80%;
    line-height: 1.5;
    word-wrap: break-word;
}

/* 機器人 (軍師) */
.bot-message {
    background: #f0f0f0;
    color: #333;
    align-self: flex-start;
    border-bottom-left-radius: 2px;
}

/* ▼▼▼ Markdown 樣式優化區 ▼▼▼ */
.bot-message p {
    margin: 5px 0; /* 讓段落不要太開 */
}
.bot-message ul, .bot-message ol {
    margin: 5px 0;
    padding-left: 25px; /* 修正列表縮排，讓它好看一點 */
}
.bot-message strong {
    color: #c2185b; /* 重點文字改成深粉紅色 */
    font-weight: 900;
}
/* ▲▲▲ 優化結束 ▲▲▲ */

/* 使用者 (你) */
.user-message {
    background: #ff4081;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 2px;
}

.input-area {
    padding: 15px;
    border-top: 1px solid #eee;
    display: flex;
    gap: 10px;
}

input {
    flex: 1;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 20px;
    outline: none;
}

button {
    background: #ff4081;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 20px;
    cursor: pointer;
    font-weight: bold;
}

button:hover { background: #e91e63; }

/* 按鈕區 */
.action-buttons {
    display: flex;
    justify-content: space-around;
    padding: 10px;
    background-color: #fce4ec;
    border-bottom: 1px solid #f0f0f0;
}

.action-btn {
    background: white;
    color: #ff4081;
    border: 1px solid #ff4081;
    border-radius: 15px;
    padding: 5px 12px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}

.action-btn:hover {
    background: #ff4081;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
  ```

</details> 

<details>
  <summary>script.js</summary>

  ```javascript
// frontend/script.js

function setMode(mode) {
    let message = "";
    if (mode === 'date') {
        message = "請幫我安排一個適合大學生的一日約會行程，風格要青春浪漫。";
    } else if (mode === 'analysis') {
        message = "我有一段跟曖昧對象的對話紀錄，請幫我分析對方對我的好感度，以及我該怎麼回覆。（請準備好，我等一下貼給你）";
    } else if (mode === 'horoscope') {
        message = "我想測今天的戀愛運勢，請給我一些幸運建議！";
    }
    const inputField = document.getElementById("user-input");
    inputField.value = message;
    sendMessage();
}

async function sendMessage() {
    const inputField = document.getElementById("user-input");
    const message = inputField.value.trim();

    if (!message) return;

    // 1. 顯示用戶訊息 (右邊, 不用 Markdown)
    addMessage(message, "user-message");
    inputField.value = ""; 

    // 2. 顯示 "思考中..." (左邊)
    const loadingId = addMessage("軍師思考中...", "bot-message");

    try {
        const response = await fetch("http://127.0.0.1:8000/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        // 3. 更新機器人回覆 (使用 Markdown 解析)
        const loadingDiv = document.querySelector(`div[data-id='${loadingId}']`);
        if (loadingDiv) {
            loadingDiv.innerHTML = marked.parse(data.reply); // ▼ 關鍵：解析 Markdown
            loadingDiv.className = "message bot-message";    // 確保樣式正確
        } else {
            addMessage(data.reply, "bot-message");
        }

    } catch (error) {
        console.error("Error:", error);
        
        // 4. 錯誤處理：直接修改原本的思考氣泡
        const loadingDiv = document.querySelector(`div[data-id='${loadingId}']`);
        
        if (loadingDiv) {
            loadingDiv.innerText = "⚠️ 軍師連線逾時 (NCKU 伺服器忙碌)，請再試一次或是按右上角重置。";
            loadingDiv.className = "message bot-message"; // 保持在左邊
            loadingDiv.style.color = "red";               // 變紅字
            loadingDiv.style.fontWeight = "bold";
        } else {
            const errorId = addMessage("⚠️ 伺服器連線錯誤", "bot-message");
            document.querySelector(`div[data-id='${errorId}']`).style.color = "red";
        }
    }
}

function addMessage(text, className) {
    const chatBox = document.getElementById("chat-box");
    const div = document.createElement("div");
    const id = Date.now();
    
    div.className = `message ${className}`;
    div.setAttribute("data-id", id);
    
    // ▼ 關鍵：如果是機器人回覆，就開啟 Markdown 解析
    if (className === 'bot-message') {
        // 為了避免一開始 "軍師思考中..." 被當成 Markdown 解析出錯，加個判斷
        if (text === "軍師思考中...") {
            div.innerText = text;
        } else {
            div.innerHTML = marked.parse(text); 
        }
    } else {
        div.innerText = text; // 使用者訊息維持純文字，避免 XSS 攻擊
    }
    
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}

document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") sendMessage();
});

async function resetChat() {
    if (!confirm("確定要清除所有對話紀錄，重新開始嗎？")) return;

    // 清空畫面
    document.getElementById("chat-box").innerHTML = 
        '<div class="message bot-message">記憶已清除！我是你的戀愛軍師，請重新提問。</div>';

    try {
        await fetch("http://127.0.0.1:8000/api/reset", { method: "POST" });
    } catch (error) {
        alert("重置失敗，請檢查後端連線");
    }
}
  ```

</details> 


###第2版

<details>
  <summary>agent_core.py</summary>

  ```python
# agent_core.py (更新版)
import requests
import json
from config import API_KEY, API_URL, MODEL_NAME, TIMEOUT
from tools import LoveTools

class LoveAgent:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        # ▼▼▼ 新增：初始化 System Prompt 和 記憶列表 ▼▼▼
        self.system_prompt = """
        你是一位專業的「戀愛軍師 AI」。你的任務是協助使用者解決戀愛煩惱。
        
        【最高指導原則】
        1. **絕對不要暴露工具失誤**：如果工具回傳「無結果」或「錯誤」，請直接用你的內建知識回答，**絕對不要說**「搜尋結果無」、「找不到資料」這種話。
        2. **語氣**：保持自信、幽默、像是大學生之間的對話。

        【人設指導原則】
        1. **拒絕盲目樂觀**：如果對方的訊息很冷淡（例如「去洗澡」、「先睡了」且沒說後續），請直接告訴使用者「這就是藉口/軟釘子」，不要硬凹成她對你有興趣。
        2. **毒舌但中肯**：可以使用一點反諷或幽默的語氣，例如：「醒醒吧兄弟，這就是洗澡卡。」
        3. **提供戰術**：點出問題後，提供「反制手段」或「停損點」。
        
        【工具使用判斷邏輯】
        1. 若使用者給出一段對話紀錄 (如 "這句話什麼意思", "他回我這個") -> 使用 `calculate_score` 分析好感度。
        2. 若使用者詢問攻略、星座、MBTI 或如何追求 -> 使用 `search_strategy` 查詢知識庫。
        3. 若使用者不知道怎麼回覆、求救 -> 使用 `get_reply_styles` 請求生成三種風格回覆。
        4. 若是尋找地點 (餐廳、電影) -> 使用 `search_web`。
        
        【回應格式】
        請輸出 JSON 格式來呼叫工具：{"tool": "工具名稱", "arg": "參數內容"}
        如果不需要工具，請直接回覆文字。
        """
        self.history = [{"role": "system", "content": self.system_prompt}]

    def reset(self):
        """新增：清除記憶功能"""
        self.history = [{"role": "system", "content": self.system_prompt}]
        return "🧹 記憶已清除，我們重新開始吧！"

    def _call_llm(self, messages):
        """內部呼叫 LLM API"""
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "temperature": 0.3
        }
        try:
            res = requests.post(API_URL, headers=self.headers, json=payload, timeout=TIMEOUT)
            if res.status_code == 200:
                return res.json()['message']['content']
            else:
                return f"Error: {res.status_code} - {res.text}"
        except Exception as e:
            return f"連線失敗: {e}"

    def _extract_json(self, text):
        try:
            cleaned = text.replace("```json", "").replace("```", "").strip()
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start != -1 and end != -1:
                return json.loads(cleaned[start:end+1])
        except: pass
        return None

    def chat(self, user_input):
        # 1. 把使用者的話加入歷史紀錄
        self.history.append({"role": "user", "content": user_input})

        # 2. 呼叫 LLM (傳送完整的歷史紀錄)
        response_text = self._call_llm(self.history)
        
        # 3. 檢查工具
        cmd = self._extract_json(response_text)
        
        if cmd and "tool" in cmd:
            tool_name = cmd["tool"]
            arg = cmd["arg"]
            tool_result = ""

            if tool_name == "calculate_score":
                tool_result = LoveTools.calculate_interest_score(arg)
            elif tool_name == "search_strategy":
                tool_result = LoveTools.search_love_strategy(arg)
            elif tool_name == "search_web":
                tool_result = LoveTools.search_web(arg)
            elif tool_name == "get_reply_styles":
                tool_result = LoveTools.generate_reply_styles(arg)
            
            # 把 AI 的第一段思考加入歷史
            self.history.append({"role": "assistant", "content": response_text})
            
            # 把工具結果加入歷史 (作為系統提示)
            tool_msg = f"【系統提示】工具回傳結果：\n{tool_result}"
            self.history.append({"role": "user", "content": tool_msg})
            
            # 再次呼叫 LLM
            final_response = self._call_llm(self.history)
            
            # 把最終回答加入歷史
            self.history.append({"role": "assistant", "content": final_response})
            return final_response
        
        else:
            # 不需要工具，直接把回答加入歷史
            self.history.append({"role": "assistant", "content": response_text})
            return response_text
  ```

</details> 

<details>
  <summary>config.py</summary>

  ```python
# config.py

# API 設定
API_KEY = "070fc5dbe1bd5a7ae8a0e2ef1b47947fd9432133f1b84a3d4a73387a96399442"
API_URL = "https://api-gateway.netdb.csie.ncku.edu.tw/api/chat"
MODEL_NAME = "gemma3:4b" # 或其他你偏好的模型

# 系統設定
TIMEOUT = 30  # API 請求超時秒數
  ```

</details> 

<details>
  <summary>main.py.html</summary>

  ```python
# main.py (更新版)
from agent_core import LoveAgent

def main():
    agent = LoveAgent()
    
    print("\n" + "="*50)
    print("💘 戀愛軍師 AI v2.1 (記憶增強版)")
    print("--------------------------------------------------")
    print("💡 指令說明：")
    print(" - 輸入一般文字：與軍師對話")
    print(" - 輸入「reset」：清除對話記憶")
    print(" - 輸入「exit」 ：離開程式")
    print("="*50)

    while True:
        try:
            user_input = input("\n[你] > ")
            
            # ▼▼▼ 處理指令 ▼▼▼
            if user_input.lower() in ["exit", "quit", "88"]:
                print("👋 祝你戀愛順利！")
                break
            
            if user_input.lower() == "reset":
                msg = agent.reset()
                print(f"🤖 {msg}")
                continue
            # ▲▲▲ 處理結束 ▲▲▲

            if not user_input.strip(): continue
            
            print("🤖 軍師思考中...", end="\r")
            reply = agent.chat(user_input)
            
            print(f"\n[軍師]：\n{reply}")
            print("-" * 30)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()
  ```

</details> 

<details>
  <summary>server.py</summary>

  ```python
# server.py (不用動，確認內容即可)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_core import LoveAgent

app = FastAPI()
agent = LoveAgent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    print(f"收到前端訊息: {req.message}")
    response_text = agent.chat(req.message)
    return {"reply": response_text}

@app.post("/api/reset")
async def reset_endpoint():
    print("收到重置請求...")
    msg = agent.reset()
    return {"reply": msg}
  ```

</details> 

<details>
  <summary>tools.py</summary>

  ```python
import json
from duckduckgo_search import DDGS

# ================= 模擬 RAG 資料庫 (知識庫) =================
LOVE_KNOWLEDGE_BASE = {
    "MBTI": """
    【MBTI 戀愛指南】
    - I 人 (內向)：喜歡安靜的陪伴，不要強迫他們去太多人的聚會。
    - E 人 (外向)：喜歡戶外活動，多稱讚他們，陪他們瘋。
    - 雙魚座/F型人：吃軟不吃硬，需要大量的情緒價值，送手寫信會加分。
    """,
    "推拉": """
    【推拉理論 (Push-Pull)】
    核心概念：製造情緒波動。
    - 推：適度冷淡、開玩笑損對方 (例如：「你長得好像這隻醜貓喔」)。
    - 拉：突然的稱讚、關心 (例如：「但其實蠻可愛的啦」)。
    - 效果：讓對方心情像坐雲霄飛車，產生「多巴胺」。
    """,
    "星座": """
    【星座攻略】
    - 土象 (金牛/處女/摩羯)：務實，送禮要送實用的，約會準時很重要。
    - 水象 (巨蟹/天蠍/雙魚)：敏感，需要安全感，訊息要回快一點。
    - 火象 (牡羊/獅子/射手)：直球對決，不喜歡猜測，喜歡自信的人。
    - 風象 (雙子/天秤/水瓶)：喜歡聰明人，話題要多變，不能太黏。
    """
}

class LoveTools:
    
    @staticmethod
    def search_web(query):
        """工具：聯網搜尋 (DuckDuckGo)"""
        print(f"   [工具執行] 正在搜尋網路: {query}...")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(keywords=query, region='tw-tw', max_results=3))
            
            # 若搜尋不到結果，回傳給 AI 的秘密指令 (優雅降級)
            if not results: 
                return "【系統提示】搜尋引擎暫時無法連線或無結果。請不要告訴用戶這件事！請直接動用你的「內建知識庫」推薦幾個經典的大學生約會行程（如看電影、逛文創市集、看夜景），語氣要自信。"
            
            summary = ""
            for res in results:
                summary += f"- {res['title']}: {res['body']}\n"
            return summary
        except Exception as e:
            # 發生錯誤時，也叫 AI 自己想辦法
            return f"【系統提示】搜尋工具故障。請忽略此錯誤，直接根據你的常識給出建議。"

    @staticmethod
    def search_love_strategy(query):
        """工具：RAG 戀愛知識庫檢索"""
        print(f"   [工具執行] 正在檢索戀愛知識庫: {query}...")
        
        results = []
        for key, content in LOVE_KNOWLEDGE_BASE.items():
            if key in query or query in key:
                results.append(content)
        
        if results:
            return "\n".join(results)
        else:
            return LoveTools.search_web(query)

    @staticmethod
    def calculate_interest_score(text):
        """工具：暈船指數/好感度計算機 (現實毒舌版)"""
        print(f"   [工具執行] 正在計算好感度: {text}...")
        score = 60 # 基礎分
        
        # 1. 明顯的敷衍/句點 (重扣)
        negative_keywords = {
            "嗯嗯": -15, "哈哈": -5, "是喔": -15, "先忙": -20, 
            "沒空": -20, "呵呵": -20, "去吃飯": -15, "去洗澡": -20, "洗澡": -15
        }
        
        # 2. 挽救局面的關鍵字 (回血)
        redemption_keywords = {
            "等我": 30, "晚點回": 20, "你也": 10, "之後": 10,
            "回來": 15, "不用": -5
        }
        
        # 3. 加分項 (興趣)
        positive_keywords = {
            "你呢": 15, "下次": 20, "這週": 20, "想去": 20, 
            "好奇": 15, "好啊": 10, "?": 5, "！": 5, "😂": 5
        }

        details = []

        # 計算扣分
        for w, point in negative_keywords.items():
            if w in text: 
                # 特殊判斷：如果是「洗澡」，檢查有沒有「挽救詞」
                if "洗澡" in w and any(r in text for r in ["等我", "晚點", "回來"]):
                    details.append(f"提及'{w}'但有承諾回來 (暫時不扣分)")
                else:
                    score += point 
                    details.append(f"偵測到句點詞 '{w}' ({point})")

        # 計算加分
        for w, point in redemption_keywords.items():
            if w in text:
                score += point
                details.append(f"偵測到挽救/正面詞 '{w}' (+{point})")
                
        for w, point in positive_keywords.items():
            if w in text:
                score += point
                details.append(f"正面情緒 '{w}' (+{point})")
        
        score = max(0, min(100, score))
        
        if score >= 85: status = "😍 暈船了！對方超愛你"
        elif score >= 60: status = "😐 有互動但還需努力"
        elif score >= 40: status = "🧊 冷淡/禮貌性回覆"
        else: status = "🥶 洗澡卡/發好人卡警報"

        return f"分數: {score}\n狀態: {status}\n分析細節: {', '.join(details) if details else '語氣平淡，無明顯關鍵字'}"

    @staticmethod
    def generate_reply_styles(scenario):
        """工具：風格回覆產生器"""
        return f"""
        【指令執行】
        使用者遇到了這個對話情境：{scenario}
        請立即生成三種不同風格的回覆建議：
        
        1. 🥶 **高冷版 (High Value)**：簡短、自信、不卑不亢，引發對方好奇。
        2. 😂 **幽默版 (Funny)**：用開玩笑化解尷尬，展現有趣靈魂。
        3. 🐶 **舔狗版 (Simp/Warm)**：極度溫柔體貼 (警告：可能顯得地位低，僅供參考)。
        
        請直接列出這三種回覆。
        """
  ```

</details> 

<details>
  <summary>index.html</summary>

  ```html
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>💘 戀愛軍師 AI</title>
    <link rel="stylesheet" href="style.css">
    
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; font-size: 24px;">💘 戀愛軍師 AI</h1>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">你的專屬感情顧問</p>
                </div>
                
                <button onclick="resetChat()" style="background: rgba(255,255,255,0.3); border: 1px solid white; color: white; padding: 6px 15px; border-radius: 20px; cursor: pointer; font-weight: bold; font-size: 14px; transition: 0.2s;">
                    🔄 重新開始
                </button>
            </div>
        </div>

        <div class="action-buttons">
            <button class="action-btn" onclick="setMode('date')">📅 安排約會</button>
            <button class="action-btn" onclick="setMode('analysis')">🧐 對話分析</button>
            <button class="action-btn" onclick="setMode('horoscope')">🔮 戀愛運勢</button>
        </div>

        <div class="chat-box" id="chat-box">
            <div class="message bot-message">你好！我是你的戀愛軍師，請點擊上方按鈕選擇服務，或直接跟我聊天！</div>
        </div>

        <div class="input-area">
            <input type="text" id="user-input" placeholder="輸入你的問題..." />
            <button class="send-btn" onclick="sendMessage()">發送</button>
        </div>
    </div>

    <script src="script.js"></script>
</body>
</html>
  ```

</details> 


<details>
  <summary>script.js</summary>

  ```javascript
// script.js (前端邏輯修復版)

function setMode(mode) {
    let message = "";
    if (mode === 'date') {
        message = "請幫我安排一個適合大學生的一日約會行程，風格要青春浪漫。";
    } else if (mode === 'analysis') {
        message = "我有一段跟曖昧對象的對話紀錄，請幫我分析對方對我的好感度，以及我該怎麼回覆。（請準備好，我等一下貼給你）";
    } else if (mode === 'horoscope') {
        message = "我想測今天的戀愛運勢，請給我一些幸運建議！";
    }
    const inputField = document.getElementById("user-input");
    inputField.value = message;
    sendMessage();
}

async function sendMessage() {
    const inputField = document.getElementById("user-input");
    const message = inputField.value.trim();

    if (!message) return;

    // 1. 顯示用戶訊息 (這是你之前消失的部分，這裡確保它會執行)
    addMessage(message, "user-message");
    
    // 清空輸入框
    inputField.value = ""; 

    // 2. 顯示 "思考中..." (左邊)
    const loadingId = addMessage("軍師思考中...", "bot-message");

    try {
        const response = await fetch("http://127.0.0.1:8000/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        // 3. 更新機器人回覆 (使用 Markdown 解析)
        const loadingDiv = document.querySelector(`div[data-id='${loadingId}']`);
        if (loadingDiv) {
            loadingDiv.innerHTML = marked.parse(data.reply); 
            loadingDiv.className = "message bot-message";    
        } else {
            addMessage(data.reply, "bot-message");
        }

    } catch (error) {
        console.error("Error:", error);
        
        // 4. 錯誤處理
        const loadingDiv = document.querySelector(`div[data-id='${loadingId}']`);
        if (loadingDiv) {
            loadingDiv.innerText = "⚠️ 軍師連線逾時，請檢查後端是否開啟 (server.py)。";
            loadingDiv.style.color = "red";
        }
    }
}

function addMessage(text, className) {
    const chatBox = document.getElementById("chat-box");
    const div = document.createElement("div");
    const id = Date.now();
    
    div.className = `message ${className}`;
    div.setAttribute("data-id", id);
    
    // ▼▼▼ 關鍵修復點 ▼▼▼
    if (className === 'bot-message') {
        // 如果是機器人，且不是思考中，就用 Markdown
        if (text === "軍師思考中...") {
            div.innerText = text;
        } else {
            div.innerHTML = marked.parse(text); 
        }
    } else {
        // ★ 如果是使用者 (user-message)，強制用純文字顯示
        // 這行如果漏掉，對話框就會是空的！
        div.innerText = text; 
    }
    // ▲▲▲ 修復結束 ▲▲▲
    
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}

document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") sendMessage();
});

async function resetChat() {
    if (!confirm("確定要清除所有對話紀錄，重新開始嗎？")) return;

    document.getElementById("chat-box").innerHTML = 
        '<div class="message bot-message">記憶已清除！我是你的戀愛軍師，請重新提問。</div>';

    try {
        await fetch("http://127.0.0.1:8000/api/reset", { method: "POST" });
    } catch (error) {
        alert("重置失敗，請檢查後端連線");
    }
}
  ```

</details> 

<details>
  <summary>style.css</summary>

  ```css
/* style.css (樣式修復版) */
body {
    background-color: #fce4ec;
    font-family: "Microsoft JhengHei", sans-serif;
    display: flex;
    justify-content: center;
    height: 100vh;
    margin: 0;
}

.chat-container {
    width: 400px;
    height: 600px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    margin-top: 50px;
}

.header {
    background: #ff4081;
    color: white;
    padding: 20px;
    text-align: center;
}

.chat-box {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    background: #fafafa;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.message {
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 80%;
    line-height: 1.5;
    word-wrap: break-word;
}

/* 機器人 (軍師) */
.bot-message {
    background: #f0f0f0;
    color: #333;
    align-self: flex-start;
    border-bottom-left-radius: 2px;
}

/* Markdown 樣式 */
.bot-message p { margin: 5px 0; }
.bot-message ul, .bot-message ol { margin: 5px 0; padding-left: 25px; }
.bot-message strong { color: #c2185b; font-weight: 900; }

/* ▼▼▼ 使用者 (你) 關鍵樣式 ▼▼▼ */
/* 確保這一段沒有被刪掉，不然你會看到白字配白底(隱形) */
.user-message {
    background: #ff4081;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 2px;
}
/* ▲▲▲ 樣式結束 ▲▲▲ */

.input-area {
    padding: 15px;
    border-top: 1px solid #eee;
    display: flex;
    gap: 10px;
}

input {
    flex: 1;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 20px;
    outline: none;
}

button {
    background: #ff4081;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 20px;
    cursor: pointer;
    font-weight: bold;
}

button:hover { background: #e91e63; }

.action-buttons {
    display: flex;
    justify-content: space-around;
    padding: 10px;
    background-color: #fce4ec;
    border-bottom: 1px solid #f0f0f0;
}

.action-btn {
    background: white;
    color: #ff4081;
    border: 1px solid #ff4081;
    border-radius: 15px;
    padding: 5px 12px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}

.action-btn:hover {
    background: #ff4081;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
  ```

</details> 
