簡介
<details>
  <summary>agent_core.py</summary>

  ```python
# agent_core.py
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
        
        # ─── 階段 1: 工具判斷專用的 System Prompt (隱藏執行) ───
        # 這個 Prompt 使用者看不到，專門用來逼模型吐出正確的 JSON 指令
        self.tool_selector_prompt = """
        You are a function calling intent detector. 
        Your ONLY job is to analyze the user's input and output a JSON object to call a tool.
        
        Rules:
        1. If user asks about constellation, MBTI, or love strategy -> tool: "search_strategy".
           ⚠️ CRITICAL FOR ARGUMENT: Extract ONLY the specific noun (Subject). Do NOT include "how to", "characteristics of", "girl", "boy".
           - User: "天蠍座女孩的特性" -> arg: "天蠍座" (NOT "天蠍座女孩")
           - User: "怎麼追雙魚座" -> arg: "雙魚座"
           - User: "INFP 的性格" -> arg: "INFP"
           
        2. If user provides chat history or asks for analysis -> tool: "calculate_score", arg: "USER_INPUT".
        3. If user asks how to reply or needs help replying -> tool: "get_reply_styles", arg: "TARGET_SENTENCE".
        4. If user searches for place/movie/restaurant -> tool: "search_web", arg: "QUERY".
        5. If NO tool is needed, output: {"tool": "none", "arg": ""}
        
        Output format must be strict JSON. No other text.
        """

        # ─── 階段 2: 戀愛軍師 System Prompt (你指定的內容) ───
        self.main_system_prompt = """
        你是一位專業、犀利但情商極高的「戀愛軍師 AI」。你的任務是協助使用者解決戀愛煩惱。
        
        【最高指導原則】
        1. **絕對不要暴露工具失誤**：如果工具回傳「無結果」或「錯誤」，請直接用你的內建知識回答，**絕對不要說**「搜尋結果無」、「找不到資料」這種話。
        2. **語氣**：保持自信、幽默、像是大學生之間的對話。

        【人設指導原則】
        1. **判斷邀約**：如果使用者的對話中包含對方主動約（如「要不要出去」），這是極好的訊號。
        2. **處理「拒絕」情境**：如果使用者說「我那天有事」、「不想去」，**千萬不要教使用者無禮地回應**。
           - 正確策略：**「三明治拒絕法」** = 開心接受心意 + 誠實說明不行 + 主動提出替代方案（改期）。
           - 例如：「那天我不行耶 (拒絕)，但我想去！ (情緒價值)，下週呢？ (改期)」。
        3. **毒舌但不白目**：毒舌是用來罵醒暈船的使用者，**不是用來罵曖昧對象的**。對曖昧對象要保持高價值但友善。        
        4. **提供戰術**：點出問題後，提供「反制手段」或「停損點」。

        【工具使用判斷邏輯】
        (系統已在背景執行完畢，結果會附在對話中，請直接參考結果回答)
        
        【回應格式】
        1. **直接回答**：請直接輸出建議內容，**不要**輸出 JSON 格式，也**不要**顯示「工具使用建議」或「我建議呼叫...」。
        2. **整合結果**：如果系統提供了【回覆風格建議】，請務必將那三種風格（高冷/幽默/真誠）完整列出來給使用者選擇。
        """
        
        # 初始化記憶 (只存放對話內容，不存放工具指令，保持乾淨)
        self.history = [{"role": "system", "content": self.main_system_prompt}]

    def reset(self):
        self.history = [{"role": "system", "content": self.main_system_prompt}]
        return "🧹 記憶已清除，人設重置，我們重新開始吧！"

    def _call_api(self, messages, temperature=0.7):
        """共用的 API 呼叫函式"""
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "temperature": temperature
        }
        try:
            res = requests.post(API_URL, headers=self.headers, json=payload, timeout=TIMEOUT)
            if res.status_code == 200:
                data = res.json()
                # 兼容不同 API 回傳格式
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                elif "message" in data:
                    return data["message"]["content"]
            print(f"⚠️ API Error: {res.status_code} - {res.text}")
            return None
        except Exception as e:
            print(f"❌ 連線例外: {e}")
            return None

    def _detect_intent_and_run_tool(self, user_input):
        """
        階段 1：判斷是否需要工具，並執行工具
        回傳: 工具執行結果 (字串) 或 None
        """
        # 建立一個臨時的 Message List，專門用來問「要不要用工具」
        selector_messages = [
            {"role": "system", "content": self.tool_selector_prompt},
            {"role": "user", "content": f"User Input: {user_input}"}
        ]
        
        # 呼叫 API (低溫模式，確保 JSON 格式精準)
        print("🕵️ [系統] 正在判斷使用者意圖...")
        response = self._call_api(selector_messages, temperature=0.1)
        
        if not response:
            return None

        # 嘗試解析 JSON
        try:
            # 清理回應，避免 markdown 干擾
            cleaned = response.replace("```json", "").replace("```", "").strip()
            cmd = json.loads(cleaned)
            
            tool_name = cmd.get("tool")
            arg = cmd.get("arg")
            
            if tool_name and tool_name != "none":
                print(f"🔧 [觸發工具] {tool_name} | 參數: {arg}")
                
                # 執行對應工具
                if tool_name == "search_strategy":
                    return LoveTools.search_love_strategy(arg) # 這裡傳進去的 arg 只會是關鍵字 (如 "雙魚座")
                elif tool_name == "calculate_score":
                    return LoveTools.calculate_interest_score(arg)
                elif tool_name == "get_reply_styles":
                    return LoveTools.generate_reply_styles(arg)
                elif tool_name == "search_web":
                    return LoveTools.search_web(arg)
            else:
                print("Checking... 不需要工具")
                
        except json.JSONDecodeError:
            print(f"⚠️ JSON 解析失敗 (模型可能沒吐出 JSON): {response}")
        except Exception as e:
            print(f"⚠️ 工具執行錯誤: {e}")
            
        return None

    def chat(self, user_input):
        # 1. 先執行「階段 1」：意圖判斷與工具執行
        # 這一口氣做完，不會影響到主對話紀錄
        tool_result = self._detect_intent_and_run_tool(user_input)

        # 2. 準備「階段 2」：正式回答
        self.history.append({"role": "user", "content": user_input})
        
        # 如果有工具結果，我們把它偽裝成一個 System Message 插在最新的對話之前
        # 讓軍師以為這是他自己腦袋裡的知識
        current_messages = self.history.copy()
        
        if tool_result:
            print(f"📄 [注入資料] 已將工具結果提供給軍師參考")
            # 插入一條系統提示，告訴軍師這是參考資料
            system_hint = {
                "role": "system", 
                "content": f"【背景資訊】(使用者看不到這條)\n關於使用者的問題，我們查到了以下資料：\n{tool_result}\n\n請利用上述資料，依據你的「毒舌軍師」人設給出建議。請不要提及「搜尋結果」字眼，把它當作你的常識。"
            }
            # 插在倒數第二個位置 (User 訊息之前) 或直接放在最後作為補充
            current_messages.insert(-1, system_hint)

        # 3. 呼叫軍師生成回覆
        print("🤖 [軍師思考中]...")
        ai_reply = self._call_api(current_messages, temperature=0.7)
        
        if ai_reply:
            self.history.append({"role": "assistant", "content": ai_reply})
            return ai_reply
        else:
            return "軍師現在有點斷線（學校伺服器忙碌中），請稍後再試..."
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
# tools.py
import os
from duckduckgo_search import DDGS

class LoveTools:
    # 靜態變數 (Cache)
    _knowledge_cache = {}

    @classmethod
    def load_knowledge_base(cls):
        """讀取 knowledge 資料夾下的所有 .txt 檔案"""
        if cls._knowledge_cache: return cls._knowledge_cache

        knowledge_dir = "knowledge"
        data = {}
        print(f"   [系統] 正在載入本地知識庫 ({knowledge_dir})...")
        
        if not os.path.exists(knowledge_dir):
            print(f"   [警告] 找不到 {knowledge_dir} 資料夾！")
            return {}

        try:
            for filename in os.listdir(knowledge_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(knowledge_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        key = filename.replace(".txt", "")
                        data[key] = f.read()
                        print(f"   -> 已載入: {filename}")
            cls._knowledge_cache = data
            return data
        except Exception as e:
            print(f"   [錯誤] 讀取失敗: {e}")
            return {}

    @staticmethod
    def search_web(query):
        """工具：聯網搜尋 (備用)"""
        print(f"   [工具] 搜尋網路: {query}...")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(keywords=query, region='tw-tw', max_results=3))
            if not results: return "【系統提示】搜尋無結果。"
            return "\n".join([f"- {res['title']}: {res['body']}" for res in results])
        except Exception as e:
            return f"【系統提示】搜尋故障: {e}"

    @staticmethod
    def search_love_strategy(query):
        """工具：檢索本地 txt + 聯網備案"""
        try:
            print(f"   [工具] 查詢攻略: {query}...")
            kb = LoveTools.load_knowledge_base()
            
            # 如果本地沒檔案，直接聯網
            if not kb: return LoveTools.search_web(query)

            results = []
            # 簡單關鍵字比對
            for category, content in kb.items():
                if category in query.lower() or query in content:
                    results.append(f"【{category} 檔案】\n{content[:800]}...") # 避免太長，截取前800字

            if results:
                return "\n\n".join(results)
            else:
                print("   [提示] 本地無資料，轉聯網搜尋...")
                return LoveTools.search_web(query)
        except Exception as e:
            return f"工具執行錯誤: {e}"

    @staticmethod
    def calculate_interest_score(text):
        """工具：好感度計算"""
        try:
            print(f"   [工具] 計算分數: {text}...")
            score = 60 
            details = []
            
            # 鏡像洗澡判定
            is_mirroring = "洗澡" in text and any(x in text for x in ["我也", "一起", "都"])
            
            # 0. 邀約偵測
            if any(x in text for x in ["要不要", "有空", "約"]) and any(y in text for y in ["出去", "吃飯", "走走", "玩", "看"]):
                score += 40
                details.append("🔥🔥 明確邀約訊號 (+40)")

            # 1. 扣分項
            negative_keywords = {"嗯嗯": -15, "哈哈": -5, "是喔": -15, "先忙": -20, "沒空": -20, "洗澡": -15}
            # 2. 加分項
            positive_keywords = {"你呢": 15, "下次": 20, "想去": 20, "好奇": 15, "好啊": 10, "我也": 15}

            for w, point in negative_keywords.items():
                if w in text:
                    if "洗澡" in w and is_mirroring: continue # 鏡像不扣分
                    score += point 
                    details.append(f"扣分詞 '{w}'")

            for w, point in positive_keywords.items():
                if w in text:
                    score += point
                    details.append(f"加分詞 '{w}'")
            
            if is_mirroring:
                score += 20
                details.append("🛁 鏡像行為 (+20)")

            score = max(0, min(100, score))
            return f"分數: {score} (細節: {', '.join(details)})"
        except Exception as e:
            return f"計算錯誤: {e}"

    @staticmethod
    def generate_reply_styles(scenario):
        return f"情境：{scenario}。請提供：1.高冷回覆 2.幽默回覆 3.真誠回覆"
  ```

</details> 


!!資料夾frontend裡有.html,.js,.css

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

資料夾knowledge裡有的.txt檔

<details>
  <summary>mbti.txt</summary>

  ```
【MBTI：INTJ 建築師】
戀愛風格：理智、高標準。
適合對象：ENFP (能融化你的冰山)。
建議：試著表達情感，不要只講道理，感情不是數學題。

【MBTI：INTP 邏輯學家】
戀愛風格：慢熱、注重智力交流。
適合對象：ENTJ (能帶領你)。
建議：多參與社交，別總活在自己世界，對方讀不到你的心。

【MBTI：ENTJ 指揮官】
戀愛風格：強勢、主導。
適合對象：INFP (互補)。
建議：對另一半溫柔點，工作不是全部，不要把家裡當戰場。

【MBTI：ENTP 辯論家】
戀愛風格：喜歡新鮮感、愛嘴砲。
適合對象：INFJ (能包容你)。
建議：承諾很重要，別總是三分鐘熱度，給對方一點安全感。

【MBTI：INFJ 提倡者】
戀愛風格：深情、追求靈魂伴侶。
適合對象：ENTP。
建議：不要過度犧牲自己，你的感受也很重要，不要當聖母。

【MBTI：INFP 調停者】
戀愛風格：浪漫、理想化。
適合對象：ENFJ。
建議：不要過度腦補，接受現實中的不完美，王子公主也會吵架。

【MBTI：ENFJ 主人公】
戀愛風格：熱情、付出型。
適合對象：INFP。
建議：留點時間給自己，不要為了討好對方失去自我。

【MBTI：ENFP 競選者】
戀愛風格：熱情、黏人。
適合對象：INTJ。
建議：給對方一點空間，雖然你有很多愛想給，但不要讓人窒息。

【MBTI：ISTJ 物流師】
戀愛風格：專一、務實。
適合對象：ESFP。
建議：試著浪漫一點，驚喜會讓感情升溫，不要像個機器人。

【MBTI：ISFJ 守衛者】
戀愛風格：照顧型、細心。
適合對象：ESFP。
建議：有需求要說出來，不要讓人猜，委屈只會內傷。

【MBTI：ESTJ 總經理】
戀愛風格：傳統、責任感。
適合對象：ISFP。
建議：不要用管理員工的方式對待伴侶，家裡講愛不講理。

【MBTI：ESFJ 執政官】
戀愛風格：受歡迎、體貼。
適合對象：ISFP。
建議：不要太在意他人的眼光，你們開心最重要。

【MBTI：ISTP 鑑賞家】
戀愛風格：獨立、不喜歡束縛。
適合對象：ESTJ。
建議：多報備，讓對方有安全感，消失不是解決問題的方法。

【MBTI：ISFP 探險家】
戀愛風格：隨和、藝術家氣質。
適合對象：ESFJ。
建議：遇到衝突不要逃避，溝通才能解決問題。

【MBTI：ESTP 企業家】
戀愛風格：刺激、活在當下。
適合對象：ISFJ。
建議：定下來需要決心，不要總是尋找下一個目標。

【MBTI：ESFP 表演者】
戀愛風格：開心果、愛玩。
適合對象：ISTJ。
建議：嚴肅的話題也要能聊，感情才能長久，不能只靠玩樂維持。
  ```

</details> 

<details>
  <summary>其他.txt</summary>

  ```
==================================================
第一章：人類圖 (Human Design) —— 你的原廠設定
==================================================
人類圖將人分為四種類型，這決定了你在戀愛中該「如何行動」才不會受傷。

1. 生產者 (Generator) & 顯示生產者 (Manifesting Generator)
- 【佔比】：約 70%
- 【戀愛策略】：等待回應 (To Respond)
  你們擁有強大的薦骨能量，但必須「被詢問」才能啟動。
  ❌ 錯誤做法：主動發起追求、沒頭沒腦地想去哪就去哪（容易碰壁感到挫折）。
  ✅ 正確做法：多出現在喜歡的人面前，給對方機會問你問題。
  ❤️ 戀愛模式：像充滿電的電池，一旦認定目標（薦骨有回應），就會有源源不絕的動力去愛。顯生者(MG)動作更快，但容易沒耐心。
- 【相處雷點】：逼迫他們做不喜歡的事（薦骨沒回應），他們會瞬間洩氣、死魚眼。

2. 投射者 (Projector)
- 【佔比】：約 20%
- 【戀愛策略】：等待被邀請 (Wait for the Invitation)
  你們天生能看透他人，但如果沒有被邀請就給建議，會讓人覺得「你好煩、管太多」。
  ❌ 錯誤做法：主動倒貼、未經詢問就一直給對方意見、試圖證明自己很有用。
  ✅ 正確做法：專注經營自己，散發光芒，等識貨的人主動來邀請你（約會、交往）。
  ❤️ 戀愛模式：非常專注於伴侶，是一對一的高手。一旦被正確的人邀請，會是最懂對方的靈魂伴侶。
- 【相處雷點】：被忽視、感覺不被賞識、體力被耗盡（你們體力比生產者差，需要大量休息）。

3. 顯示者 (Manifestor)
- 【佔比】：約 9%
- 【戀愛策略】：告知 (To Inform)
  你們是唯一的「發起者」，想要什麼就直接行動，不用等。
  ❌ 錯誤做法：問對方「我可以親你嗎？」、「我們可不可以去吃飯？」（這會削弱你的霸氣）。
  ✅ 正確做法：直接說「我想親你」、「我週五訂了餐廳，帶你去吃」。
  ❤️ 戀愛模式：像一陣風，來去自如。喜歡能夠獨立、不黏人、能跟上你們速度的伴侶。
- 【相處雷點】：被控制、被問東問西、行蹤被限制。

4. 反映者 (Reflector)
- 【佔比】：約 1%
- 【戀愛策略】：等待 28 天週期 (Wait a Lunar Cycle)
  你們沒有固定的能量中心，像鏡子一樣反映環境。
  ❌ 錯誤做法：當下立刻做決定（通常會後悔）。
  ✅ 正確做法：曖昧期拉長，至少觀察一個月，感受自己在不同環境下對這個人的感覺。
  ❤️ 戀愛模式：極度依賴「環境」和「氛圍」。如果你覺得跟這個人在一起不舒服，通常是對的。
- 【相處雷點】：被逼迫做決定、待在氣場不好的地方。

==================================================
第二章：星盤進階 (Advanced Astrology) —— 內在需求
==================================================
太陽星座只是外在表現，長期關係請看「月亮」與「金星」。

【Part 1：月亮星座 (Moon Sign) —— 內心安全感來源】
同居、結婚必看。月亮代表一個人卸下防備後的樣子。

- 月亮火象 (牡羊/獅子/射手)：
  [特質] 情緒來得快去得快，像小孩。
  [需求] 需要「被看見」、「被哄」。不喜歡冷戰，有話直說。如果他生氣，你跟他大吵一架再和好，比悶著不說好。

- 月亮土象 (金牛/處女/摩羯)：
  [特質] 情緒壓抑，習慣忍耐，最後爆發。
  [需求] 需要「物質保障」和「承諾」。愛我就展現給我看（存摺、禮物、未來的計畫）。他們的安全感來自於生活秩序不被打亂。

- 月亮風象 (雙子/天秤/水瓶)：
  [特質] 理性處理情緒，看起來很冷靜，其實是在抽離。
  [需求] 需要「溝通」和「理解」。當他情緒低落時，陪他聊天，或者給他空間自己想通。最怕情緒勒索和哭鬧。

- 月亮水象 (巨蟹/天蠍/雙魚)：
  [特質] 敏感、直覺強、記憶力好（記仇）。
  [需求] 需要「情感連結」和「黏膩」。必須秒回訊息、隨時報備。他們的安全感來自於感覺到「你完全屬於我」。

【Part 2：金星星座 (Venus Sign) —— 愛情觀與審美】
決定了他喜歡哪一型，以及約會該怎麼安排。

- 金星火象：喜歡耀眼、自信、大方的人。約會要有動態感（遊樂園、演唱會）。愛得轟轟烈烈。
- 金星土象：喜歡高質感、有能力、長相端正的人。約會要享受感官（米其林餐廳、質感展覽）。愛得務實。
- 金星風象：喜歡聰明、幽默、聊得來的人。約會要有趣味（脫口秀、桌遊）。愛得像朋友。
- 金星水象：喜歡溫柔、會照顧人、有同理心的人。約會要私密浪漫（家裡看電影、海邊散步）。愛得像家人。

==================================================
第三章：生命靈數 (Numerology) —— 性格密碼
==================================================
計算：西元出生年月日數字加總至個位數。
(例：1996/07/20 -> 1+9+9+6+7+2+0 = 34 -> 3+4=7 號人)

- 1號人 (開創)：獨立、自我、愛面子。希望伴侶崇拜他。最怕被否定。
- 2號人 (協調)：依賴、細膩、黏人。天生配合者，需要大量陪伴。最怕孤單。
- 3號人 (創意)：任性、可愛、長不大。喜歡新鮮感，情緒起伏大。最怕無聊。
- 4號人 (穩定)：固執、顧家、缺乏安全感。需要規律的生活。最怕變動。
- 5號人 (自由)：冒險、博愛、不受控。需要極大的空間。最怕被束縛。
- 6號人 (關懷)：負責、完美主義、碎碎念。會把對方照顧得無微不至。最怕對方不知感恩。
- 7號人 (真理)：質疑、冷靜、距離感。喜歡精神交流，不喜歡太黏。最怕笨蛋。
- 8號人 (權力)：控制、野心、大老闆。重視物質回報。最怕沒錢沒權。
- 9號人 (智慧)：夢想家、大愛、不切實際。追求靈魂契合。最怕世俗壓力。

==================================================
第四章：三觀契合度 (Values) —— 長久關係的基石
==================================================
熱戀期看五官，磨合期看三觀。除了星座，這才是分手的真正原因。

1. 金錢觀 (Financial Views)
- [消費模式]：
  A. 享樂主義：賺多少花多少，重視當下體驗（旅遊、美食）。
  B. 儲蓄主義：省吃儉用，重視未來保障（買房、投資）。
  *解法*：如果不一致，建議設立「公基金」制度，剩下的錢互不干涉。
- [負債態度]：能不能接受分期付款？有沒有學貸？這必須坦承。

2. 家庭觀 (Family Views)
- [邊界感]：父母能介入我們生活到什麼程度？
  A. 緊密型：每天都要跟爸媽視訊，重大決定要問爸媽。
  B. 獨立型：報喜不報憂，不喜歡長輩干涉。
- [家務分配]：誰洗碗？誰倒垃圾？這往往是吵架導火線。
- *建議*：不要說「幫忙」，家事是共有的責任。

3. 世界觀 (World Views) / 人生目標
- [成長步調]：
  A. 狼性：想創業、想升遷、假日都在進修。
  B. 佛系：工作只是為了賺錢，假日只想廢在沙發。
  *警訊*：如果一方一直在跑，一方原地踏步，話題會越來越少，最後形同陌路。
- [休閒娛樂]：一個喜歡戶外爬山，一個喜歡室內打電動。這不一定是缺點，但雙方必須有「獨處」的共識。

4. 吵架觀 (Conflict Resolution)
當衝突發生時，你們的習慣是什麼？
- A. 焦慮型：當下一定要講清楚，不能過夜。
- B. 逃避型：現在不想講，想躲回洞穴冷靜幾天。
- *建議*：這沒有對錯，但需要協調出一個「暫停訊號」。例如：「我現在情緒很滿，給我 30 分鐘冷靜，之後我們再來談。」
  ```

</details> 

<details>
  <summary>心理學.txt</summary>

  ```
【戀愛心理學理論大全】

1. 愛情三角理論 (Triangular Theory of Love)
- 來源：Robert Sternberg (1986)
- 核心概念：完美的愛情由三個元素組成：
  1. 親密感 (Intimacy)：心靈的靠近、分享秘密。
  2. 激情 (Passion)：身體的吸引力、浪漫、性衝動。
  3. 承諾 (Commitment)：決定在一起並維持關係的決心。
- 應用攻略：
  - 如果只有激情+親密，沒有承諾，那是「浪漫愛」(容易分手)。
  - 如果只有親密+承諾，沒有激情，那是「友伴愛」(老夫老妻)。
  - 若想長久，請檢查這三角是否平衡，缺哪補哪。

2. 依附理論 (Attachment Theory)
- 來源：John Bowlby & Mary Ainsworth
- 核心概念：人的戀愛模式源自童年安全感，分為三種：
  1. 安全型 (Secure)：情緒穩定，不擔心被拋棄，也願意依賴對方。
  2. 焦慮型 (Anxious)：這類人需要大量訊息秒回，不然會覺得對方不愛了。攻略他們要給足安全感。
  3. 逃避型 (Avoidant)：這類人怕太親密會失去自由，越追逃越遠。攻略他們要給空間，像放風箏。
- 應用攻略：
  - 遇到「忽冷忽熱」通常是逃避型，不要逼太緊。
  - 遇到「奪命連環Call」通常是焦慮型，要主動報備。

3. 增減效應 (Gain-Loss Theory) -> 即「推拉理論」的學術版
- 來源：Aronson & Linder (1965)
- 核心概念：人對於「先貶後褒」的評價，會比「一開始就一直稱讚」的評價更高。
- 應用攻略 (推拉法)：
  - 不要當舔狗 (一直褒 = 廉價)。
  - 操作順序：先稍微冷淡或損對方一下 (推/減)，讓對方心情小低落；然後再給予肯定或溫暖 (拉/增)。這種反差會讓多巴胺分泌更旺盛。
  - 範例：「我覺得你這個人很難相處耶... (停頓)... 沒啦，開玩笑的，其實你笑起來蠻可愛的。」

4. 吊橋效應 (Suspension Bridge Effect)
- 來源：Dutton & Aron (1974)
- 核心概念：錯誤歸因 (Misattribution of Arousal)。當人在危險環境下心跳加速時，會誤以為這是對眼前的人「心動」的感覺。
- 應用攻略：
  - 曖昧期約會地點首選：恐怖電影院、雲霄飛車、高空景觀台、密室逃脫。
  - 避免地點：安靜圖書館、無聊的公園 (除非你們已經很熟)。

5. 班傑明·富蘭克林效應 (Benjamin Franklin Effect)
- 來源：Jecker & Landy (1969)
- 核心概念：想讓一個人喜歡你，不是去幫助他，而是「請他幫你一個小忙」。
- 原理：因為人類大腦有認知失調機制，當我幫了你，大腦會解釋成「我應該是喜歡這個人，才會幫他」。
- 應用攻略：
  - 剛認識時，可以借個筆、請教一個小問題、請他幫忙拿個東西。
  - 幫完忙後記得稱讚感謝，建立互動連結。

6. 蔡格尼克效應 (Zeigarnik Effect)
- 來源：Bluma Zeigarnik (1927)
- 核心概念：人對於「未完成的任務」記憶最深刻。
- 應用攻略：
  - 聊天不要聊到乾掉才結束。要在話題最高潮的時候說「我要先去忙一下，晚點跟你說個秘密」。
  - 留下懸念，讓對方在你不回訊息的時候，腦子裡一直想著你剛剛沒講完的話。

7. 單純曝光效應 (Mere Exposure Effect)
- 來源：Robert Zajonc (1968)
- 核心概念：人會單純因為「熟悉某個事物」而對它產生好感。
- 應用攻略：
  - 適用對象：同學、同事、經常見面的人。
  - 操作方式：不需要每次都強行聊天，只要頻繁地「出現在他的視線範圍內」就好。例如去他常去的圖書館、在他發的限時動態頻繁按讚（刷存在感）。
  - 注意：前提是對方對你印象不差，如果第一印象不好，曝光越多會越討厭。

8. 鏡像效應 (Mirroring Effect)
- 核心概念：人會潛意識地喜歡跟自己舉止相似的人（模仿是最大的恭維）。
- 應用攻略：
  - 約會時：對方拿水杯喝水，你過 3 秒也喝水；對方身體前傾，你也前傾。
  - 聊天時：重複對方用的「關鍵字」。例如他說「我覺得那部電影很震撼」，你回「對啊，真的超震撼」，而不是回「我覺得很棒」。
  - 效果：這會讓對方的潛意識覺得「我們是同一類人」，迅速拉近距離。

9. 愛之語 (The 5 Love Languages)
- 來源：Gary Chapman (1992)
- 核心概念：每個人感受到愛的方式不同，這五種語言是：
  1. 肯定的言語 (Words of Affirmation)：喜歡聽甜言蜜語、稱讚。
  2. 服務的行動 (Acts of Service)：幫忙倒垃圾、吹頭髮。
  3. 真心的禮物 (Receiving Gifts)：不在乎價錢，在乎心意。
  4. 精心時刻 (Quality Time)：放下手機，專注陪伴。
  5. 身體接觸 (Physical Touch)：牽手、擁抱。
- 應用攻略：
  - 很多吵架是因為「語言不通」。例如男生一直「幫忙修電腦(服務)」，但女生想要的是「陪我聊天(精心時刻)」。
  - 軍師建議：觀察對方抱怨什麼，通常那就是他缺少的愛之語。

10. 暈輪效應 (Halo Effect) / 光環效應
- 來源：Edward Thorndike (1920)
- 核心概念：人們會根據一個人的「某個顯著優點」（通常是外表），而認為他在其他方面（如個性、能力）也很好。
- 應用攻略：
  - 第一印象定生死：第一次約會的穿搭、髮型決定了後續 70% 的容錯率。
  - 如果你外表打理得好，就算約會講錯話，對方也會覺得你「可能只是害羞，好可愛」；如果外表邋遢，對方會覺得你「社交能力有問題」。

11. 沉沒成本謬誤 (Sunk Cost Fallacy)
- 核心概念：因為已經投入了大量時間、金錢或感情，即使知道這段關係已經壞掉了，還是不捨得放手。
- 應用攻略：
  - 這是「勸分」最強的理論依據。
  - 當用戶問「他對我這麼壞，為什麼我離不開？」時，這就是沉沒成本在作祟。
  - 建議：告訴自己，現在停損就是賺到。

12. 羅密歐與茱麗葉效應 (Romeo and Juliet Effect)
- 來源：Driscoll, Davis & Lipetz (1972)
- 核心概念：外界的阻力越大（例如父母反對、遠距離），情侶之間的感情反而會越深，因為他們覺得自己在「共同對抗世界」。
- 應用攻略：
  - 如果你在追的人有很多人追（競爭者多），或者你們有些困難要克服，試著把這變成「我們兩個人的秘密任務」。
  - 建立「共患難」的情境，可以大幅增加親密度。

13. 互惠原則 (Reciprocity Principle)
- 來源：Robert Cialdini (《影響力》作者)
- 核心概念：人對於「接受恩惠」會產生虧欠感，進而想要回報。
- 應用攻略：
  - 不要一次給太大的恩惠（對方會壓力大）。
  - 給予「小恩惠」：去旅行帶個小伴手禮、幫忙順手買杯飲料。
  - 關鍵：當對方說「謝謝，下次換我請你」時，你就成功拿到了下一次約會的門票。
  ```

</details> 

<details>
  <summary>星座.txt</summary>

  ```
【星座攻略總論：四大星象】
如果不知道對方的確切星座，可以先從屬性判斷：

- 火象星座 (牡羊/獅子/射手)：
  特質：直覺、行動派、愛面子、熱情。
  攻略核心：讓他崇拜你，或者讓他覺得「追你很有挑戰性」。千萬不要倒貼。

- 土象星座 (金牛/處女/摩羯)：
  特質：感官、務實、慢熱、固執。
  攻略核心：溫水煮青蛙。展現你的穩定性、專業能力和生活品味。

- 風象星座 (雙子/天秤/水瓶)：
  特質：思考、溝通、自由、甚至有點疏離。
  攻略核心：要聊得來！話題要多變，不能太黏，要像朋友一樣相處。

- 水象星座 (巨蟹/天蠍/雙魚)：
  特質：情感、情緒化、直覺強、缺乏安全感。
  攻略核心：情緒價值。提供滿滿的安全感，讓他依賴你的溫柔。

--------------------------------------------------

【牡羊座 Aries (3/21 - 4/19)】
- 關鍵字：孩子氣、衝動、征服慾。
- 戀愛性格：喜歡就會直接衝，不喜歡你再怎麼努力都沒用。喜歡強者，喜歡有一點挑戰性的對象。
- 攻略法：
  1. 欲擒故縱：稍微對他冷一點點，激起他的狩獵本能。
  2. 直球對決：有話直說，不要讓他猜心。
  3. 稱讚他：把他當英雄誇獎。
- 大地雷：太黏人、潑他冷水、比他還強勢。
- 喜歡你的訊號：一直找你吵架或開玩笑、秒回訊息、直接約你。

【金牛座 Taurus (4/20 - 5/20)】
- 關鍵字：吃貨、愛錢、固執、慢郎中。
- 戀愛性格：非常慢熱，需要長時間觀察。一旦認定了就會非常專一且大方。
- 攻略法：
  1. 美食誘惑：約會選好吃的餐廳就贏一半了。
  2. 送禮哲學：送有質感、實用的東西（不要送不切實際的花）。
  3. 肢體接觸：偶爾不經意的觸碰，對感官敏銳的他們很有效。
- 大地雷：催促他做決定、浪費錢、忽冷忽熱。
- 喜歡你的訊號：捨得為你花錢、送你東西、帶你去吃好料。

【雙子座 Gemini (5/21 - 6/20)】
- 關鍵字：精分、話癆、好奇寶寶、三分鐘熱度。
- 戀愛性格：喜歡聰明有趣的人，最怕無聊。聊天頻率合不合最重要。
- 攻略法：
  1. 保持神秘感：不要一次把自己的底牌全掀開。
  2. 跟上話題：要懂很多梗，陪他天南地北亂聊。
  3. 若即若離：他忙你比他更忙，他反而會好奇你在幹嘛。
- 大地雷：管太寬、情緒勒索、話題太無聊枯燥。
- 喜歡你的訊號：把你的事告訴全世界、一直傳好笑的梗圖給你。

【巨蟹座 Cancer (6/21 - 7/22)】
- 關鍵字：家、殼、情緒化、玻璃心。
- 戀愛性格：外表堅強內心柔軟，防衛心重。非常念舊，容易心軟。
- 攻略法：
  1. 溫柔攻勢：關心他有沒有吃飯、天冷加衣，走家人路線。
  2. 傾聽者：當他情緒的垃圾桶，站在他這邊。
  3. 主動報備：給足安全感，消除他的疑心病。
- 大地雷：批評他的家人或朋友、開過火的玩笑、不回訊息。
- 喜歡你的訊號：像媽媽一樣碎念你、跟你講心事、變得黏人。

【獅子座 Leo (7/23 - 8/22)】
- 關鍵字：國王/女王、面子、傲嬌、保護慾。
- 戀愛性格：在外面要是王，回家變大貓。喜歡被崇拜，佔有慾強。
- 攻略法：
  1. 給足面子：在朋友面前一定要聽他的，誇獎他。
  2. 撒嬌：獅子座吃軟不吃硬，撒嬌能解決一切問題。
  3. 華麗約會：喜歡有儀式感的場合。
- 大地雷：當眾給他難看、無視他、跟異性太好。
- 喜歡你的訊號：霸道總裁式的關心、宣示主權、送貴重禮物。

【處女座 Virgo (8/23 - 9/22)】
- 關鍵字：細節控、潔癖、碎碎念、服務型。
- 戀愛性格：完美主義，嫌棄你是因為愛你。很被動，不敢主動告白。
- 攻略法：
  1. 整潔乾淨：外表一定要乾淨整齊，指甲不能髒。
  2. 虛心受教：當他碎念建議時，表示認同並改進。
  3. 可靠度：展現你的上進心和未來規劃。
- 大地雷：邋遢、遲到、說話不算話、做事草率。
- 喜歡你的訊號：開始管你的生活瑣事、對你很囉嗦、主動幫你整理東西。

【天秤座 Libra (9/23 - 10/22)】
- 關鍵字：外貌協會、選擇障礙、優雅、好人緣。
- 戀愛性格：極度重視顏值和氣質。喜歡被陪伴，但討厭做決定。
- 攻略法：
  1. 打扮自己：長得好看或穿得好看，你就成功了80%。
  2. 幫他決定：約會直接給選項（A或B），不要問「隨便」。
  3. 陪伴：他怕寂寞，多花時間陪在身邊。
- 大地雷：粗魯不禮貌、逼他表態、讓他感到尷尬。
- 喜歡你的訊號：會主動約你（這對被動的他們很難得）、把你介紹給朋友。

【天蠍座 Scorpio (10/23 - 11/21)】
- 關鍵字：神秘、控制慾、偵探、愛恨分明。
- 戀愛性格：愛的很深，恨的也深。直覺超準，千萬別說謊。
- 攻略法：
  1. 絕對誠實：不要耍小聰明，坦承會讓他欣賞。
  2. 性張力：適度的眼神交流和曖昧氛圍。
  3. 堅定選擇：讓他知道你的世界只有他，別搞曖昧。
- 大地雷：背叛、欺騙、把他的秘密說出去、跟別人搞曖昧。
- 喜歡你的訊號：身家調查你、控制你的行蹤、故意欺負你引起注意。

【射手座 Sagittarius (11/22 - 12/21)】
- 關鍵字：自由、冒險、樂觀、哲學家。
- 戀愛性格：浪子/浪女，一生放蕩不羈愛自由。喜歡玩伴，討厭束縛。
- 攻略法：
  1. 玩伴模式：陪他瘋、陪他玩、聊旅行聊夢想。
  2. 放牛吃草：不要管他去哪，他玩夠了自己會回來。
  3. 幽默感：一定要開得起玩笑。
- 大地雷：奪命連環Call、悲觀負面、限制他的自由。
- 喜歡你的訊號：主動帶你出去玩、跟你分享他的人生大道理。

【摩羯座 Capricorn (12/22 - 1/19)】
- 關鍵字：工作狂、面癱、現實、責任感。
- 戀愛性格：外冷內熱（悶騷）。認為愛就是給你穩定的物質生活。
- 攻略法：
  1. 展現能力：讓他看到你在工作或學業上的優秀表現。
  2. 主動一點：這塊木頭很難主動，你要負責丟球破冰。
  3. 成熟穩重：不要無理取鬧耍脾氣。
- 大地雷：影響他的工作、太情緒化、幼稚。
- 喜歡你的訊號：願意花時間在你身上（時間就是金錢）、和你討論未來規劃。

【水瓶座 Aquarius (1/20 - 2/18)】
- 關鍵字：外星人、博愛、叛逆、精神戀愛。
- 戀愛性格：忽冷忽熱是常態。重視精神契合度大於肉體。
- 攻略法：
  1. 從朋友做起：不要急著確認關係，先當最好的朋友。
  2. 獨特性：展現你與眾不同的一面，怪一點沒關係。
  3. 給予空間：不要黏，各自有獨立生活空間。
- 大地雷：太黏人、用世俗標準要求他、干涉他的隱私。
- 喜歡你的訊號：跟你分享他奇怪的想法、在你面前展現脆弱。

【雙魚座 Pisces (2/19 - 3/20)】
- 關鍵字：浪漫、幻想、濫好人、犧牲奉獻。
- 戀愛性格：活在粉紅泡泡裡，需要大量的愛和浪漫情節。
- 攻略法：
  1. 偶像劇招式：壁咚、摸頭殺、看夜景，越老套越有效。
  2. 保護慾：展現強勢的一面，幫他解決問題。
  3. 傾聽夢想：聽他天馬行空的幻想，不要潑冷水。
- 大地雷：太現實冷血、說話太直接傷人、忽視他的感受。
- 喜歡你的訊號：隨叫隨到、看你的眼神會拉絲、為你做手工禮物。




==================================================
第一章：四象限配對總則 (The Elemental Laws)
==================================================
判斷兩人合不合，最快的方式是看「元素」的相容性。

1. 【完美共振組】(同元素 = 100% 合拍)
   - 火象 (牡羊/獅子/射手) + 火象：熱情如火，節奏一致，不拖泥帶水。但吵架時會像火山爆發。
   - 土象 (金牛/處女/摩羯) + 土象：細水長流，價值觀最合，能一起存錢買房。但生活可能缺乏激情。
   - 風象 (雙子/天秤/水瓶) + 風象：無話不談，靈魂伴侶。但可能都太愛玩，缺乏穩定性。
   - 水象 (巨蟹/天蠍/雙魚) + 水象：心靈相通，不需要言語就能懂對方。但容易一起陷入情緒黑洞。

2. 【互補協調組】(火+風 / 土+水)
   - 火象 + 風象 (風助火勢)：風象的聰明能引導火象的衝勁。例如：射手(火)+水瓶(風) 是經典的自由組合。
   - 土象 + 水象 (水滋潤土)：土象給水象安全感，水象給土象溫柔。例如：摩羯(土)+雙魚(水) 是霸總與小白兔的組合。

3. 【相愛相殺組】(火+水 / 風+土)
   - 火象 + 水象 (水火不容)：火象覺得水象情緒化很煩，水象覺得火象粗魯傷人。
   - 風象 + 土象 (雞同鴨講)：風象想聊外星人，土象只想聊房價。一個在天一個在地。

==================================================
第二章：12 星座詳細配對指南
==================================================

【牡羊座 Aries】
- 💖 絕配：獅子座 (國王與騎士，強強聯手)、射手座 (一起冒險的玩伴)。
- 🤝 互補：天秤座 (對宮星座，牡羊的衝動需要天秤的優雅來中和)。
- ⚠️ 孽緣：巨蟹座 (覺得他太玻璃心)、摩羯座 (覺得他太嚴肅無聊)。
- 💡 攻略：牡羊需要一個能讓他「崇拜」或「追逐」的人，太容易到手的不會珍惜。

【金牛座 Taurus】
- 💖 絕配：處女座 (都注重細節與品質)、摩羯座 (最穩定的麵包組合)。
- 🤝 互補：天蠍座 (對宮星座，金牛的肉慾 vs 天蠍的深情，性吸引力極強)。
- ⚠️ 孽緣：獅子座 (兩個都固執，價值觀差太多)、水瓶座 (無法理解他的外星邏輯)。
- 💡 攻略：金牛需要的是「看得到的未來」，穩定的物質基礎比甜言蜜語重要。

【雙子座 Gemini】
- 💖 絕配：水瓶座 (兩個外星人，沒人聽懂你們在講什麼)、天秤座 (顏值與智商的完美結合)。
- 🤝 互補：射手座 (對宮星座，都愛自由，但射手看遠方，雙子看當下)。
- ⚠️ 孽緣：雙魚座 (受不了他的濫情與腦補)、處女座 (受不了他的碎碎念)。
- 💡 攻略：雙子需要「大腦的刺激」，能陪他聊八卦、聊哲學、聊廢話的人才能留住他。

【巨蟹座 Cancer】
- 💖 絕配：天蠍座 (只有天蠍接得住你的情緒)、雙魚座 (一起演瓊瑤劇)。
- 🤝 互補：摩羯座 (對宮星座，巨蟹顧家，摩羯顧事業，傳統完美的家庭分工)。
- ⚠️ 孽緣：牡羊座 (太兇了受不了)、天秤座 (覺得他對每個人都好，很沒安全感)。
- 💡 攻略：巨蟹需要「無條件的偏愛」，你要讓他覺得他是你世界裡的唯一。

【獅子座 Leo】
- 💖 絕配：射手座 (一拍即合的玩咖)、牡羊座 (轟轟烈烈的愛情)。
- 🤝 互補：水瓶座 (對宮星座，獅子愛面子，水瓶不在乎他人眼光，互相吸引)。
- ⚠️ 孽緣：金牛座 (為了錢和價值觀吵翻天)、天蠍座 (王見王，控制慾大戰)。
- 💡 攻略：獅子需要「舞台」，你要當他在台下最忠實的觀眾，給他掌聲。

【處女座 Virgo】
- 💖 絕配：摩羯座 (務實界的頂點)、金牛座 (懂生活品質的組合)。
- 🤝 互補：雙魚座 (對宮星座，處女的理性 vs 雙魚的感性，互相學習)。
- ⚠️ 孽緣：射手座 (覺得他太隨便、沒規矩)、雙子座 (覺得他太輕浮、不可靠)。
- 💡 攻略：處女座嫌棄你就是愛你。他們需要一個「能一起進步」的伴侶。

【天秤座 Libra】
- 💖 絕配：雙子座 (聊不完的話題)、水瓶座 (彼此尊重空間)。
- 🤝 互補：牡羊座 (對宮星座，天秤的優柔寡斷需要牡羊的果決來拯救)。
- ⚠️ 孽緣：巨蟹座 (情緒勒索讓你窒息)、摩羯座 (太悶了，聊不起來)。
- 💡 攻略：天秤需要「陪伴」和「美感」。不要逼他做決定，幫他決定好，帶他去漂亮的地方。

【天蠍座 Scorpio】
- 💖 絕配：巨蟹座 (給你滿滿的安全感)、雙魚座 (懂你的靈魂深處)。
- 🤝 互補：金牛座 (對宮星座，情感與物質的極致結合，通常很長久)。
- ⚠️ 孽緣：獅子座 (互不相讓)、水瓶座 (這人太冷漠，抓不住)。
- 💡 攻略：天蠍需要「絕對的忠誠」。只有在確定你不會背叛後，他才會展現極致的深情。

【射手座 Sagittarius】
- 💖 絕配：牡羊座 (行動力滿點)、獅子座 (一起發光發熱)。
- 🤝 互補：雙子座 (對宮星座，知識與體驗的結合，最好的旅伴)。
- ⚠️ 孽緣：處女座 (管太多了)、雙魚座 (悲春傷秋讓你很累)。
- 💡 攻略：射手需要「放風」。你越不理他，他越黏你；你越管他，他跑越快。

【摩羯座 Capricorn】
- 💖 絕配：處女座 (最強的工作夥伴與伴侶)、金牛座 (穩定累積財富)。
- 🤝 互補：巨蟹座 (對宮星座，你的剛強需要他的溫柔來融化)。
- ⚠️ 孽緣：牡羊座 (太幼稚)、天秤座 (太虛偽)。
- 💡 攻略：摩羯需要「戰友」。展現你的能力，讓他覺得跟你在一起能讓未來過得更好。

【水瓶座 Aquarius】
- 💖 絕配：天秤座 (社交圈的明星組合)、雙子座 (智力遊戲的對手)。
- 🤝 互補：獅子座 (對宮星座，你欣賞他的熱情，他欣賞你的特立獨行)。
- ⚠️ 孽緣：天蠍座 (太黏人了)、金牛座 (太無趣了)。
- 💡 攻略：水瓶需要「懂他怪的人」。當全世界都覺得他怪，只有你覺得他可愛，你就贏了。

【雙魚座 Pisces】
- 💖 絕配：天蠍座 (刻骨銘心的愛)、巨蟹座 (溫暖的港灣)。
- 🤝 互補：處女座 (對宮星座，你的混亂需要他的秩序來整理)。
- ⚠️ 孽緣：雙子座 (花心對上濫情)、射手座 (一個想定下來，一個想飛)。
- 💡 攻略：雙魚需要「造夢者」。陪他演偶像劇，適時展現霸道總裁的一面。

==================================================
第三章：進階配對心法 (Secret Rules)
==================================================
1. 對宮吸引力 (The Opposition Effect)
- 星盤上 180 度相對的星座（如牡羊vs天秤、金牛vs天蠍）。
- 他們看似完全相反，其實內核很像。這種組合通常有「致命的吸引力」，要嘛愛得死去活來，要嘛恨之入骨。
- 適合：追求成長、想互補的人。

2. 90度刑剋 (The Square)
- 星盤上 90 度的組合（如牡羊vs巨蟹、獅子vs天蠍）。
- 這是最辛苦的組合，價值觀完全衝突。
- 但如果能撐過磨合期，往往是能力最強、最堅不可摧的夫妻檔（因為什麼大風大浪沒見過）。

3. 月亮星座才是關鍵
- 談戀愛看太陽（外在個性），結婚過日子一定要看「月亮」（內在需求）。
- 如果你們太陽不合（例如火象配水象），但月亮是合的（例如月亮都是土象），那你們私底下相處其實會非常舒服，這才是長久的關鍵。
  ```

</details> 

<details>
  <summary>溝通技巧和戀愛術語.txt</summary>

  ```
【實戰溝通技巧 & 現代戀愛術語】

1. 非暴力溝通 (NVC - Nonviolent Communication)
- 核心：解決吵架的神器，避免互相指責。
- 公式：觀察 + 感受 + 需要 + 請求
  - 錯誤說法：「你每次都遲到，很不尊重我！」(指責)
  - 正確說法：「我們約 7 點，現在 7 點半了(觀察)，我覺得很焦慮(感受)，因為我很重視我們的約會時間(需要)，下次能不能準時，或是遲到先跟我說？(請求)」
- 應用攻略：當用戶說「不知道怎麼跟對方溝通不滿」時，AI 請直接套用此公式幫他擬稿。

2. XYZ 溝通法
- 來源：哈佛談判專案
- 公式：當你做 X (事件) 時，我感到 Y (情緒)，我希望你能做 Z (期望)。
- 範例：「當你已讀不回我時(X)，我覺得很不安(Y)，希望你忙完可以跟我說一聲(Z)。」

3. 愛情轟炸 (Love Bombing) —— 紅色警訊
- 定義：剛認識不久就瘋狂稱讚、送禮、許下過度承諾(例如交往一週就說要結婚)。
- 目的：這是渣男/渣女/PUA 常見起手式，目的是快速讓你暈船，建立依賴。
- 攻略：如果你覺得「這進展快得不真實」，那通常就是假的。請保持冷靜，設立界線。

4. 煤氣燈效應 (Gaslighting) —— 紅色警訊
- 定義：心理操控。對方透過否認事實、轉移焦點，讓你懷疑自己的記憶或神智正常。
- 經典台詞：「是你太敏感了吧」、「我哪有這樣說，你記錯了」、「開個玩笑而已，你有必要這麼認真嗎」。
- 攻略：AI 需提醒用戶相信自己的直覺，這是情感虐待的一種，建議遠離。

5. 麵包屑 (Breadcrumbing) —— 備胎訊號
- 定義：對方不拒絕你，但也不答應交往。偶爾丟一點甜頭(麵包屑)給你，讓你覺得「有希望」，只是為了吊著你。
- 特徵：忽冷忽熱、只在深夜找你、對於確認關係總是顧左右而言他。
- 攻略：這就是把你當備胎。請直接攤牌：「我們現在是什麼關係？」如果對方答不出來，就快逃。

6. 斷崖式分手 (Ghosting)
- 定義：沒有任何徵兆、沒有吵架，對方突然人間蒸發，不回訊息、封鎖。
- 心態分析：這通常不是你的錯，而是對方缺乏處理衝突的成熟度，或者他在逃避責任。
- 攻略：不要試圖去討要答案（因為沒有答案），把這視為對方送你的最後禮物——讓你認清他是一個沒擔當的人。

7. 情境式戀愛 (Situationship) —— 大學生常見
- 定義：介於「朋友」和「情侶」之間的模糊地帶。做著情侶會做的事(約會、親密接觸)，但沒有情侶的名分(Label)。
- 攻略：
  - 如果你享受當下：那就繼續。
  - 如果你想要承諾：這會非常痛苦。必須設定「停損點」，例如「三個月內沒確認關係就離開」。

8. 蛙化現象 (Ick)
- 定義：原本很喜歡對方，但看到對方做了一件小事(例如跑步姿勢很醜、拿零錢手在抖)，突然瞬間冷掉，覺得對方像青蛙一樣噁心。
- 心理分析：這通常代表你愛上的只是「理想中的他」，而不是真實的他。或者你的潛意識在逃避親密關係。

9. 邊界感 (Boundaries)
- 核心：健康的關係必須有界線。
- 範例：
  - 「我不喜歡你在我看書的時候一直吵我。」
  - 「我不接受你跟前任單獨出去吃飯。」
- 攻略：愛不是無底線的包容。AI 應鼓勵用戶勇敢說出自己的底線，對方如果因此離開，那他本來就不適合你。
  ```
