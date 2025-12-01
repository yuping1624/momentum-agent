"""
Mind Flow App - Streamlit 界面
只負責顯示和用戶交互，核心邏輯在 brain.py
"""
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import pandas as pd
import datetime
import os
import time
import html
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from brain import create_mind_flow_brain, load_user_profile

# --- 安全關鍵字（Guardrails） ---
SAFETY_KEYWORDS = [
    # English
    "suicide",
    "kill myself",
    "want to die",
    "want to end it all",
    "end my life",
    "self-harm",
    "self harm",
    # Chinese
    "自殺",
    "想死",
    "不想活了",
    "活不下去",
    "想結束一切",
    "傷害自己",
]

SAFETY_MESSAGE = (
    "⚠️ 我注意到你提到可能與自我傷害或生命安全有關的內容。\n\n"
    "我是一個 AI，沒有醫療或心理專業資格，也無法在緊急狀況中提供即時協助。\n\n"
    "👉 如果你有**立即的危險**，請立刻聯絡你所在地的緊急電話（例如 911），\n"
    "或撥打當地的自殺防治／心理諮詢專線，並尋求家人、朋友或信任的人陪伴你。\n\n"
    "你值得被好好對待，也值得被真正看見和幫助。"
)


# --- 共用訊息 / 調試渲染函數 ---
def render_message(msg):
    """根據訊息角色，將 User / Agent 分別顯示在左右兩側，並加上色塊。"""
    if isinstance(msg, HumanMessage):
        # 使用者在左側（藍色氣泡）
        left, right = st.columns([3, 1])
        with left:
            st.markdown(
                f"""
                <div class="mf-msg mf-user">
                    <span class="mf-avatar">👤</span>
                    <span class="mf-text">{msg.content}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif isinstance(msg, AIMessage):
        # Agent 在右側（綠色氣泡）
        left, right = st.columns([1, 3])
        with right:
            st.markdown(
                f"""
                <div class="mf-msg mf-agent">
                    <span class="mf-avatar">🤖</span>
                    <span class="mf-text">{msg.content}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_supervisor_cot(result):
    """在畫面上顯示 Supervisor 的推理過程（Chain of Thought）與路由結果（顯示在 Agent 回覆上方）。"""
    if not isinstance(result, dict):
        return
    reasoning = result.get("reasoning")
    debug_info = result.get("debug_info")
    if not reasoning and not debug_info:
        return

    # 轉義 HTML 並將換行符顯示為 <br>，確保 Step 1/2/3 分行清楚
    reasoning_html = ""
    if reasoning:
        escaped = html.escape(reasoning)
        reasoning_html = escaped.replace("\r\n", "\n").replace("\n", "<br>")

    debug_html = ""
    if debug_info:
        debug_html = html.escape(debug_info)

    # 全寬度顯示一個灰色的推理卡片
    st.markdown(
        f"""
        <div class="mf-cot">
            <div class="mf-cot-title">💭 Supervisor Chain of Thought</div>
            {f"<div class='mf-cot-debug'>{debug_html}</div>" if debug_html else ""}
            {f"<div class='mf-cot-body'>{reasoning_html}</div>" if reasoning_html else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- 1. 初始化與設定 ---
load_dotenv()
st.set_page_config(page_title="Mind Flow", page_icon="🧠", layout="wide")

# CSS 優化 (讓介面更乾淨 + 訊息色塊樣式)
st.markdown("""
<style>
    .stChatMessage { font-family: 'Helvetica', sans-serif; }
    .stButton button { border-radius: 20px; }

    /* 共用訊息卡片樣式 */
    .mf-msg {
        padding: 0.6rem 0.8rem;
        border-radius: 10px;
        margin: 0.2rem 0;
        display: inline-block;
        max-width: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .mf-avatar {
        margin-right: 0.35rem;
        font-size: 0.95rem;
    }
    .mf-text {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .mf-user {
        background-color: #e3f2fd;  /* 淡藍 */
        color: #0d47a1;
    }
    .mf-agent {
        background-color: #e8f5e9;  /* 淡綠 */
        color: #1b5e20;
    }

    /* Supervisor Chain-of-Thought 卡片（灰色） */
    .mf-cot {
        background-color: #f5f5f5;  /* 淺灰 */
        border-left: 4px solid #9e9e9e;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0 0.2rem 0;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        font-size: 0.85rem;
    }
    .mf-cot-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #424242;
    }
    .mf-cot-debug {
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: #616161;
    }
    .mf-cot-body {
        margin: 0;
        white-space: pre-wrap;
        font-family: Menlo, Monaco, Consolas, "Courier New", monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 側邊欄：設定與數據儀表板 ---
with st.sidebar:
    st.header("⚙️ Mind Flow Engine")
    
    # API Key 管理 (優先級: 環境變數 > Secrets > 手動輸入)
    # 1. 優先從環境變數讀取 (通過 load_dotenv() 從 .env 文件加載)
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # 2. 如果環境變數沒有，嘗試從 Streamlit Secrets 讀取
    if not api_key:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
        except StreamlitSecretNotFoundError:
            pass  # secrets.toml 不存在，繼續下一步
    
    # 3. 如果都沒有，使用手動輸入
    if not api_key:
        api_key = st.text_input("Google API Key", type="password", help="請輸入 Gemini API Key")

    st.divider()
    
    # 調試：顯示 user_profile 狀態
    if st.checkbox("🔍 顯示調試信息", False):
        user_profile = load_user_profile()
        st.write("**User Profile 狀態:**")
        st.json(user_profile)
        if st.button("🗑️ 清除對話記錄（測試用）"):
            if "messages" in st.session_state:
                del st.session_state.messages
            st.rerun()
    
    st.subheader("🧭 你的導航系統")
    
    # 從 JSON 文件加載用戶配置文件
    user_profile = load_user_profile()
    
    if user_profile.get("vision"):
        st.markdown(f"**🔭 願景:** {user_profile['vision']}")
        st.markdown(f"**⚙️ 系統:** {user_profile['system']}")
        st.info("💡 Starter 會根據你的當前狀態動態生成微行動建議")
    else:
        st.warning("尚未建立系統。請與 Strategist 互動以設定你的 12 週願景！")

    st.divider()
    
    # 初始化資料庫 (Session State 模擬)
    if "journal_db" not in st.session_state:
        st.session_state.journal_db = pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note"])

    st.subheader("📊 Flow Journal")
    if not st.session_state.journal_db.empty:
        # 顯示最近 5 筆
        st.dataframe(st.session_state.journal_db.tail(5), hide_index=True)
        # 簡單趨勢圖
        st.line_chart(st.session_state.journal_db["Energy"])
    else:
        st.info("尚無數據，完成一次行動後會自動記錄。")

if not api_key:
    st.warning("請先輸入 API Key 才能啟動 Mind Flow。")
    st.stop()

# --- 3. 初始化大腦 ---
# 創建更新日記的回調函數
def update_journal(mood: str, energy: int, note: str):
    """更新日記資料庫的回調函數"""
    new_entry = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Mood": mood,
        "Energy": energy,
        "Note": note
    }
    st.session_state.journal_db = pd.concat(
        [st.session_state.journal_db, pd.DataFrame([new_entry])], 
        ignore_index=True
    )

# 使用 session_state 來緩存大腦實例，避免每次重新創建
if "mind_flow_app" not in st.session_state:
    st.session_state.mind_flow_app = create_mind_flow_brain(
        api_key=api_key,
        model="gemini-2.0-flash",
        update_callback=update_journal
    )

# --- 4. 使用者介面 (UX) ---

st.title("🧠 Mind Flow")
st.caption("From Anxiety to Action: Your AI Companion for Executive Function.")

# --- 快速建議按鈕（放在主畫面最上方，接近標題） ---
suggestions = ["🎯 幫我拆解目標", "😫 我現在好焦慮", "🐢 我想動但動不了", "✅ 我完成了！幫我紀錄"]
cols = st.columns(4)
selected_prompt = None
for i, suggestion in enumerate(suggestions):
    with cols[i]:
        if st.button(suggestion):
            selected_prompt = suggestion

# 初始化對話
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # 根據 user_profile 的狀態決定使用哪個 Agent
    from brain import get_strategist_greeting, get_returning_user_greeting
    # 從 JSON 文件加載用戶配置文件
    user_profile = load_user_profile()
    
    # 檢查是否已完成 onboarding（system 已設置）
    if user_profile.get("system"):
        # 老用戶：直接使用 Starter（啟動）或 Healer（關心）
        # 預設使用 Starter（啟動模式），如果需要 Healer 可以改為 "healer"
        with st.spinner("🚀 Starter 正在準備問候（老用戶模式）..."):
            greeting_response = get_returning_user_greeting(
                api_key=api_key, 
                model="gemini-2.0-flash",
                plan_state=user_profile,
                agent_type="starter"  # 或 "healer" 用於關心模式
            )
    else:
        # 新用戶或未完成 onboarding：使用 Strategist
        with st.spinner("🧠 Strategist 正在準備問候..."):
            greeting_response = get_strategist_greeting(
                api_key=api_key, 
                model="gemini-2.0-flash",
                plan_state=user_profile
            )
    
    st.session_state.messages.append(greeting_response)

# --- 輸入區（放在主畫面最下方） ---

# 先取得使用者輸入
user_input = st.chat_input("告訴我你現在的狀態...")

# 決定本輪實際要送給 Agent 的文字：優先使用 chat_input，其次是上方快速按鈕
prompt = user_input or selected_prompt

# 輸入處理：只更新狀態（messages、sidebar 等），真正的顯示統一在下方歷史訊息迴圈處理
if prompt:
    # 1. 加入 User Message
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    # 1.5 安全檢查：自我傷害／生命危險關鍵字（硬守門）
    lowered = prompt.lower()
    if any(keyword in lowered for keyword in SAFETY_KEYWORDS):
        # 直接用固定模板回覆，不進入大腦／不調用任何工具
        safety_ai_message = AIMessage(content=SAFETY_MESSAGE)
        st.session_state.messages.append(safety_ai_message)
        st.warning("⚠️ 安全守門機制已觸發，此輪對話不會進入 Mind Flow 大腦。")
    else:
        # 2. 執行 Agent（使用輕量提示，而不是整頁模糊的 spinner）
        status = st.empty()
        status.markdown("⏳ Mind Flow 團隊正在協作中...")
        result = st.session_state.mind_flow_app.invoke({"messages": st.session_state.messages})
        response = result["messages"][-1]
        status.empty()
        
        # 3. 加入 AI Response
        st.session_state.messages.append(response)

        # 3.5 如果有 Supervisor 推理結果，這一輪更新後在下方一起渲染
        st.session_state.last_supervisor_result = result

        # 4. 如果有 Tool Call，顯示成功提示
        has_set_full_plan = False
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # 檢查是哪種工具被調用
            for tool_call in response.tool_calls:
                tool_name = getattr(tool_call, 'name', None) or (tool_call.get('name') if isinstance(tool_call, dict) else None)
                if tool_name == "save_journal_entry":
                    st.toast("✨ 日記已寫入資料庫！查看側邊欄數據。", icon="✅")
                elif tool_name == "set_full_plan":
                    has_set_full_plan = True
                    st.toast("✨ 計劃已建立！查看側邊欄導航系統。", icon="🎯")
        # 5. 只要本輪任一工具調用了 set_full_plan（無論 demo 或一般對話），立刻 rerun 更新側邊欄
        if has_set_full_plan:
            st.rerun()

# 顯示歷史訊息（包含本輪新增的 user/agent）
for idx, msg in enumerate(st.session_state.messages):
    render_message(msg)
    # 在每個 Agent 回覆之後，如果有對應的 Supervisor 推理結果，就顯示在該回覆底下
    if (
        isinstance(msg, AIMessage)
        and "last_supervisor_result" in st.session_state
        and idx == len(st.session_state.messages) - 1  # 目前只對最後一輪顯示 CoT
    ):
        render_supervisor_cot(st.session_state.last_supervisor_result)
