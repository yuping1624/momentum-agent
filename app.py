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


# --- RLHF 回饋紀錄函數 ---
def log_feedback(user_input: str, agent_response: str, rating: int):
    """
    將使用者回饋記錄到 CSV 檔案。
    rating: 1 = 👍, -1 = 👎
    """
    os.makedirs("data", exist_ok=True)
    feedback_path = os.path.join("data", "feedback_ratings.csv")
    # 清理文本，避免在 CSV 中產生多行；將換行轉成可讀的 "\n"
    def _clean(text: str) -> str:
        if not isinstance(text, str):
            return str(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.replace("\n", "\\n")

    new_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": _clean(user_input),
        "agent_response": _clean(agent_response),
        "rating": rating,
    }
    if os.path.exists(feedback_path) and os.path.getsize(feedback_path) > 0:
        try:
            df_existing = pd.read_csv(feedback_path)
        except pd.errors.EmptyDataError:
            df_existing = pd.DataFrame(columns=["timestamp", "user_input", "agent_response", "rating"])
        df = pd.concat([df_existing, pd.DataFrame([new_record])], ignore_index=True)
    else:
        df = pd.DataFrame([new_record])
    df.to_csv(feedback_path, index=False, encoding="utf-8")


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
        # Agent 在右側（綠色氣泡，整體靠右對齊）
        left, right = st.columns([1, 3])
        with right:
            st.markdown(
                f"""
                <div class="mf-agent-wrap">
                    <div class="mf-msg mf-agent">
                        <span class="mf-avatar">🤖</span>
                        <span class="mf-text">{msg.content}</span>
                    </div>
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


# --- 數據持久化與讀取函數 ---
MIND_FLOW_DB_PATH = os.path.join("data", "mind_flow_db.csv")

def load_mind_flow_db():
    """從 CSV 文件加載日記數據庫"""
    os.makedirs("data", exist_ok=True)
    db_path = MIND_FLOW_DB_PATH
    if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
        try:
            df = pd.read_csv(db_path)
            # 確保必要的列存在
            required_cols = ["Timestamp", "Mood", "Energy", "Note", "type"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None if col != "type" else "JOURNAL_LOG"
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note", "type"])
    return pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note", "type"])

def save_to_mind_flow_db(timestamp: str, mood: str, energy: int, note: str):
    """保存日記條目到 CSV 文件（帶錯誤處理和重試機制）"""
    os.makedirs("data", exist_ok=True)
    db_path = MIND_FLOW_DB_PATH
    
    new_entry = {
        "Timestamp": timestamp,
        "Mood": mood,
        "Energy": energy,
        "Note": note,
        "type": "JOURNAL_LOG"
    }
    
    try:
        if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
            try:
                df_existing = pd.read_csv(db_path)
                # 檢查是否已存在相同的記錄（避免重複）
                if not df_existing.empty:
                    duplicate = (
                        (df_existing["Timestamp"] == timestamp) & 
                        (df_existing["Mood"] == mood) & 
                        (df_existing["Energy"] == energy) &
                        (df_existing.get("Note", "") == note)
                    ).any()
                    if duplicate:
                        return df_existing  # 已存在，不重複保存
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                df_existing = pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note", "type"])
            df = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([new_entry])
        
        # 保存到 CSV，確保編碼正確
        df.to_csv(db_path, index=False, encoding="utf-8")
        return df
    except Exception as e:
        # 如果保存失敗，記錄錯誤但不中斷程序
        print(f"⚠️ 保存日記到 CSV 失敗: {e}")
        # 嘗試創建一個備份文件
        try:
            backup_path = db_path.replace(".csv", "_backup.csv")
            df.to_csv(backup_path, index=False, encoding="utf-8")
            print(f"✅ 已保存到備份文件: {backup_path}")
        except:
            pass
        return None

def calculate_dashboard_metrics():
    """計算儀表板指標"""
    df = load_mind_flow_db()
    
    # Total Actions: 類型為 'JOURNAL_LOG' 的行數
    journal_logs = df[df["type"] == "JOURNAL_LOG"] if "type" in df.columns else df
    total_actions = len(journal_logs)
    
    # Avg Energy: Energy 列的平均值（處理缺失值）
    if "Energy" in journal_logs.columns and not journal_logs.empty:
        energy_values = pd.to_numeric(journal_logs["Energy"], errors="coerce")
        avg_energy = energy_values.mean()
        avg_energy = round(avg_energy, 1) if not pd.isna(avg_energy) else 0.0
    else:
        avg_energy = 0.0
    
    # Current Streak: 最近 7 天的日誌數量
    if "Timestamp" in journal_logs.columns and not journal_logs.empty:
        try:
            journal_logs["Timestamp"] = pd.to_datetime(journal_logs["Timestamp"], errors="coerce")
            seven_days_ago = datetime.datetime.now() - datetime.timedelta(days=7)
            recent_logs = journal_logs[journal_logs["Timestamp"] >= seven_days_ago]
            current_streak = len(recent_logs)
        except:
            current_streak = 0
    else:
        current_streak = 0
    
    return {
        "total_actions": total_actions,
        "avg_energy": avg_energy,
        "current_streak": current_streak
    }


load_dotenv()
st.set_page_config(page_title="Mind Flow", page_icon="🧠", layout="wide")

# 初始化日記資料庫 (Session State) - 從 CSV 加載或創建新的
if "journal_db" not in st.session_state:
    df_loaded = load_mind_flow_db()
    journal_logs = df_loaded[df_loaded["type"] == "JOURNAL_LOG"] if "type" in df_loaded.columns else df_loaded
    # 只保留必要的列給 session_state（不包含 type）
    if not journal_logs.empty:
        st.session_state.journal_db = journal_logs[["Timestamp", "Mood", "Energy", "Note"]].copy()
        # 設置標記，表示數據已從 CSV 加載
        st.session_state.journal_db_loaded_from_csv = True
    else:
        st.session_state.journal_db = pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note"])
        st.session_state.journal_db_loaded_from_csv = False

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
    .mf-agent-wrap {
        width: 100%;
        text-align: right;  /* 讓 Agent 氣泡整體靠右對齊 */
    }
    .mf-agent {
        background-color: #e8f5e9;  /* 淡綠 */
        color: #1b5e20;
        display: inline-block;      /* 配合 wrap 做靠右排列 */
        text-align: left;            /* 框內文字靠左對齊 */
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
    # === Quantified Self Dashboard (頂部 Metrics) ===
    st.header("📊 Quantified Self")
    metrics = calculate_dashboard_metrics()
    
    # 使用 columns 顯示三個關鍵指標
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Actions", metrics["total_actions"])
    with col2:
        st.metric("Avg Energy", f"{metrics['avg_energy']:.1f}")
    with col3:
        st.metric("7-Day Streak", metrics["current_streak"])
    
    st.divider()
    
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
    
    # 數據狀態顯示
    journal_count = len(st.session_state.journal_db) if "journal_db" in st.session_state else 0
    if journal_count > 0:
        st.caption(f"📝 已加載 {journal_count} 筆日記記錄")
    else:
        st.caption("📝 尚無日記記錄")
    
    # 調試：顯示 user_profile 狀態
    if st.checkbox("🔍 顯示調試信息", False):
        user_profile = load_user_profile()
        st.write("**User Profile 狀態:**")
        st.json(user_profile)
        st.write("**日記數據狀態:**")
        st.write(f"- Session State 記錄數: {len(st.session_state.journal_db)}")
        df_csv = load_mind_flow_db()
        csv_logs = df_csv[df_csv["type"] == "JOURNAL_LOG"] if "type" in df_csv.columns else df_csv
        st.write(f"- CSV 文件記錄數: {len(csv_logs)}")
        st.write(f"- CSV 文件路徑: {MIND_FLOW_DB_PATH}")
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

if not api_key:
    st.warning("請先輸入 API Key 才能啟動 Mind Flow。")
    st.stop()

# --- 3. 初始化大腦 ---
# 創建更新日記的回調函數
def update_journal(mood: str, energy: int, note: str):
    """更新日記資料庫的回調函數（同時更新 session_state 和 CSV）"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    new_entry = {
        "Timestamp": timestamp,
        "Mood": mood,
        "Energy": energy,
        "Note": note
    }
    # 更新 session_state
    st.session_state.journal_db = pd.concat(
        [st.session_state.journal_db, pd.DataFrame([new_entry])], 
        ignore_index=True
    )
    # 同時保存到 CSV 文件（確保持久化）
    try:
        result = save_to_mind_flow_db(timestamp, mood, energy, note)
        if result is None:
            # 保存失敗，但已經更新了 session_state，所以至少這次會話中可見
            print(f"⚠️ 警告：日記條目已更新到 session_state，但保存到 CSV 失敗")
    except Exception as e:
        print(f"⚠️ 保存日記時發生錯誤: {e}")
        # 即使保存失敗，也繼續執行，至少 session_state 中有數據

# 使用 session_state 來緩存大腦實例，避免每次重新創建
if "mind_flow_app" not in st.session_state:
    st.session_state.mind_flow_app = create_mind_flow_brain(
        api_key=api_key,
        model="gemini-2.0-flash",
        update_callback=update_journal
    )

# --- 4. 使用者介面 (UX) ---

st.title("Mind Flow")
st.caption("From Anxiety to Action: Your AI Companion for Executive Function.")

# 建立主分頁：對話 / 儀表板
tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard"])

with tab_chat:
    # --- 快速建議按鈕（放在 Chat 分頁頂部） ---
    suggestions = ["🎯 幫我拆解目標", "😫 我現在好焦慮", "🐢 我想動但動不了", "✅ 我完成了！幫我紀錄"]
    cols = st.columns(4)
    selected_prompt = None
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion):
                selected_prompt = suggestion

    # 建立一個容器用來承載歷史訊息，確保它始終顯示在輸入框上方
    history_container = st.container()

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

    # --- 輸入區（Chat 分頁底部） ---

    # 先取得使用者輸入
    user_input = st.chat_input("告訴我你現在的狀態...")

    # 決定本輪實際要送給 Agent 的文字：優先使用 chat_input，其次是上方快速按鈕
    prompt = user_input or selected_prompt

    # 輸入處理：只更新狀態（messages、sidebar 等）
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

            # 3.5 記錄本輪 Supervisor 推理結果，供渲染時對應到這個回覆
            if "cot_history" not in st.session_state:
                st.session_state.cot_history = []
            # 目前這個 AI 回覆的索引就是最後一個
            ai_index = len(st.session_state.messages) - 1
            st.session_state.cot_history.append({"idx": ai_index, "result": result})
            
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

    # 在 history_container 中渲染歷史訊息與 RLHF 回饋，確保它們總是在輸入框上方
    with history_container:
        # 顯示歷史訊息（包含本輪新增的 user/agent），並記錄最後一組 User / Agent 對
        last_user_msg = None
        last_agent_msg = None
        for idx, msg in enumerate(st.session_state.messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg
                render_message(msg)
            elif isinstance(msg, AIMessage):
                last_agent_msg = msg
                # 先顯示對應這個 idx 的 Supervisor 推理結果（灰色方塊在回覆上方）
                if "cot_history" in st.session_state:
                    for entry in st.session_state.cot_history:
                        if entry.get("idx") == idx:
                            render_supervisor_cot(entry.get("result"))
                            break
                # 再顯示 Agent 回覆本身
                render_message(msg)
            else:
                # 其他類型訊息（保險起見）
                render_message(msg)

        # RLHF 回饋按鈕（只對最後一個 Agent 回覆顯示，貼在 Agent 區塊右下角）
        if last_user_msg is not None and last_agent_msg is not None:
            # 依據當前最後一個 Agent 訊息的 index，維護對應的 feedback 狀態，避免跨輪殘留
            if "feedback_status" not in st.session_state:
                st.session_state.feedback_status = {}
            if "last_agent_index" not in st.session_state:
                st.session_state.last_agent_index = None

            # 如果這一輪的最後一個 Agent index 跟前一輪不同，重置這一輪的狀態
            current_agent_index = len(st.session_state.messages) - 1
            if st.session_state.last_agent_index != current_agent_index:
                st.session_state.last_agent_index = current_agent_index
                st.session_state.feedback_status[current_agent_index] = None

            current_status = st.session_state.feedback_status.get(current_agent_index)

            # 佈局：三欄，前兩欄留白，最後兩欄是緊鄰的讚 / 倒讚按鈕（更靠近在一起）
            spacer, col_up, col_down = st.columns([6, 1, 1])
            with col_up:
                if st.button("👍", key=f"feedback_up_{current_agent_index}"):
                    log_feedback(last_user_msg.content, last_agent_msg.content, rating=1)
                    st.session_state.feedback_status[current_agent_index] = "up"
                    current_status = "up"
            with col_down:
                if st.button("👎", key=f"feedback_down_{current_agent_index}"):
                    log_feedback(last_user_msg.content, last_agent_msg.content, rating=-1)
                    st.session_state.feedback_status[current_agent_index] = "down"
                    current_status = "down"

            # 小提示文字緊貼在按鈕下方，只針對這一輪的 Agent 顯示
            if current_status == "up":
                st.caption("🙏 已記錄這次回覆為「有幫助」")
            elif current_status == "down":
                st.caption("📥 已記錄這次回覆為「不太好」")

with tab_dashboard:
    st.header("📊 Flow Journal Dashboard")
    
    # 從 CSV 加載完整數據（包含歷史記錄）
    df_full = load_mind_flow_db()
    journal_logs = df_full[df_full["type"] == "JOURNAL_LOG"] if "type" in df_full.columns else df_full
    
    if not journal_logs.empty:
        # 頂部統計卡片
        st.subheader("📈 Overview")
        overview_cols = st.columns(4)
        with overview_cols[0]:
            st.metric("Total Entries", len(journal_logs))
        with overview_cols[1]:
            if "Energy" in journal_logs.columns:
                energy_vals = pd.to_numeric(journal_logs["Energy"], errors="coerce")
                avg_energy = energy_vals.mean()
                st.metric("Avg Energy", f"{avg_energy:.1f}" if not pd.isna(avg_energy) else "N/A")
            else:
                st.metric("Avg Energy", "N/A")
        with overview_cols[2]:
            if "Mood" in journal_logs.columns:
                most_common_mood = journal_logs["Mood"].mode()[0] if not journal_logs["Mood"].mode().empty else "N/A"
                st.metric("Most Common Mood", most_common_mood)
            else:
                st.metric("Most Common Mood", "N/A")
        with overview_cols[3]:
            if "Timestamp" in journal_logs.columns:
                try:
                    journal_logs["Timestamp"] = pd.to_datetime(journal_logs["Timestamp"], errors="coerce")
                    seven_days_ago = datetime.datetime.now() - datetime.timedelta(days=7)
                    recent_count = len(journal_logs[journal_logs["Timestamp"] >= seven_days_ago])
                    st.metric("Last 7 Days", recent_count)
                except:
                    st.metric("Last 7 Days", "N/A")
            else:
                st.metric("Last 7 Days", "N/A")
        
        st.divider()
        
        # 能量趨勢圖表
        st.subheader("📉 Energy Trend")
        if "Timestamp" in journal_logs.columns and "Energy" in journal_logs.columns:
            try:
                chart_data = journal_logs[["Timestamp", "Energy"]].copy()
                chart_data["Timestamp"] = pd.to_datetime(chart_data["Timestamp"], errors="coerce")
                chart_data["Energy"] = pd.to_numeric(chart_data["Energy"], errors="coerce")
                chart_data = chart_data.dropna().sort_values("Timestamp")
                if not chart_data.empty:
                    st.line_chart(chart_data.set_index("Timestamp")["Energy"], width='stretch')
                else:
                    st.info("能量數據不足，無法顯示趨勢圖。")
            except Exception as e:
                st.warning(f"無法繪製趨勢圖：{str(e)}")
        else:
            st.info("缺少必要的數據列（Timestamp 或 Energy）。")
        
        st.divider()
        
        # 最近日記記錄表格
        st.subheader("📝 Recent Journal Entries")
        display_cols = ["Timestamp", "Mood", "Energy", "Note"]
        available_cols = [col for col in display_cols if col in journal_logs.columns]
        if available_cols:
            recent_data = journal_logs[available_cols].tail(20)
            st.dataframe(recent_data, hide_index=True, width='stretch')
        else:
            st.info("沒有可顯示的數據列。")
    else:
        st.info("💡 尚無日記數據，完成一次行動後會自動記錄。")
        st.markdown("""
        **如何開始：**
        - 與 Agent 對話並完成一次行動
        - Agent 會自動記錄你的狀態（Mood, Energy, Note）
        - 數據會顯示在這裡
        """)

