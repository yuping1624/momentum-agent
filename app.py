import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import pandas as pd
import datetime
import os
from dotenv import load_dotenv

# LangChain & LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

# --- 1. 初始化與設定 ---
load_dotenv()
st.set_page_config(page_title="Mind Flow", page_icon="🧠", layout="wide")

# CSS 優化 (讓介面更乾淨)
st.markdown("""
<style>
    .stChatMessage { font-family: 'Helvetica', sans-serif; }
    .stButton button { border-radius: 20px; }
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

# --- 3. 定義工具 (Tools) ---
@tool
def save_journal_entry(mood: str, energy: int, note: str):
    """
    [Architect 專用] 將使用者的狀態存入資料庫。
    Args:
        mood: 使用者情緒關鍵字 (如: Anxious, Flowing, Stuck)
        energy: 自評能量指數 (1-10)
        note: 對話摘要或行動紀錄
    """
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
    return f"✅ 已紀錄：Mood={mood}, Energy={energy}"

# --- 4. 定義 Agent Prompts (核心靈魂) ---

# 1. 策略家 (新增)：負責拆解目標
strategist_prompt = """
You are 'The Strategist', a 12-Week Year planner.
Your Goal: Clarify vague goals into actionable plans.

Guidelines:
1. **Refuse Vague Goals:** If user says "I want to lose weight", ask "What is the specific metric?"
2. **The 12-Week Mindset:** Focus on what can be done THIS week to move the needle.
3. **Outcome:** End with a clear plan, then hand over to 'The Starter' to execute the first step.
"""

# 2. 療癒者 (Gemini 風格)：負責安撫情緒
healer_prompt = """
You are 'The Healer', a companion with deep emotional intelligence (Gemini-style).
Your Goal: Make the user feel 100% understood and safe.

**Core Personality Guidelines:**
1. **Pacing over Solving:** Do NOT offer solutions in your first response. Spend 100% of the effort on validation.
   - Bad: "You feel sad. Do this."
   - Good: "It sounds like a really heavy day. That feeling of wanting to move but being stuck is incredibly exhausting."
2. **Rich Vocabulary:** Use nuanced emotional words (e.g., "frazzled", "weighed down", "scattered").
3. **Tentative Tone:** Use phrases like "I wonder if...", "It makes sense that...", "Perhaps...".
4. **The "We" Perspective:** Always use "We". "Let's sit with this feeling."
"""

# 3. 啟動者：負責打破慣性
starter_prompt = """
You are 'The Starter', an Atomic Habits coach.
Your Goal: Convert intent into a tiny, undeniable action (Micro-step).

Guidelines:
1. **Be Concise:** Keep response SHORT (max 3 sentences). Long text = cognitive load.
2. **Negotiate Down:** If user hesitates, lower the bar. "Can't run? Just put on shoes."
3. **Action First:** Don't talk about feelings anymore. Talk about motion.
"""

# 4. 架構師：負責紀錄與優化
architect_prompt = """
You are 'The Architect'.
Your Goal: Log the data and optimize the environment.

Guidelines:
1. **Always Log:** You MUST use the 'save_journal_entry' tool to save the session data.
2. **Environment Design:** Give ONE tip to optimize their physical space for next time (e.g., "Put the yoga mat by the bed").
3. **Reinforce Identity:** Tell them: "You are the type of person who takes action."
"""

# --- 5. LangGraph 建構 ---

# 初始化 LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    next_step: str

# Nodes
def strategist_node(state):
    messages = [SystemMessage(content=strategist_prompt)] + state["messages"]
    return {"messages": [llm.invoke(messages)], "next_step": "END"}

def healer_node(state):
    messages = [SystemMessage(content=healer_prompt)] + state["messages"]
    return {"messages": [llm.invoke(messages)], "next_step": "END"}

def starter_node(state):
    messages = [SystemMessage(content=starter_prompt)] + state["messages"]
    return {"messages": [llm.invoke(messages)], "next_step": "END"}

def architect_node(state):
    # Architect 綁定工具
    llm_with_tools = llm.bind_tools([save_journal_entry])
    messages = [SystemMessage(content=architect_prompt)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)], "next_step": "END"}

# Supervisor (Router)
def supervisor_node(state):
    router_prompt = """
    Analyze the user's latest message and Intent. Route to the best specialist:
    
    1. 'STRATEGIST': User wants to set goals, plan, or is confused about what to do.
    2. 'HEALER': User is sad, anxious, tired, stuck, guilt-ridden, or venting.
    3. 'STARTER': User is emotionally okay but lazy/procrastinating, or ready to act.
    4. 'ARCHITECT': User has finished a task, wants to log progress, or says "I did it".
    
    Return ONLY the word: STRATEGIST, HEALER, STARTER, or ARCHITECT.
    """
    messages = [SystemMessage(content=router_prompt)] + state["messages"]
    response = llm.invoke(messages).content.upper()
    
    if "STRATEGIST" in response: return {"next_step": "strategist"}
    elif "HEALER" in response: return {"next_step": "healer"}
    elif "STARTER" in response: return {"next_step": "starter"}
    elif "ARCHITECT" in response: return {"next_step": "architect"}
    else: return {"next_step": "healer"} # Default fallback

# Graph Definition
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("strategist", strategist_node)
workflow.add_node("healer", healer_node)
workflow.add_node("starter", starter_node)
workflow.add_node("architect", architect_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges("supervisor", lambda x: x["next_step"], 
                               {"strategist": "strategist", "healer": "healer", 
                                "starter": "starter", "architect": "architect"})

workflow.add_edge("strategist", END)
workflow.add_edge("healer", END)
workflow.add_edge("starter", END)
workflow.add_edge("architect", END)

app = workflow.compile()

# --- 6. 使用者介面 (UX) ---

st.title("🧠 Mind Flow")
st.caption("From Anxiety to Action: Your AI Companion for Executive Function.")

# 初始化對話
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # 主動問候 (Proactive Greeting)
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "早安。新的一天開始了。你想先設定今天的『核心目標』(Strategist)，還是覺得有點沒動力(Healer)？"
    elif 12 <= current_hour < 18:
        greeting = "午後好。今天進度如何？如果卡住了，我們隨時可以微調目標。"
    else:
        greeting = "晚上好。今天辛苦了。要不要花 2 分鐘結算一下今天的狀態 (Architect)？"
    
    st.session_state.messages.append(AIMessage(content=greeting))

# 顯示歷史訊息
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# 建議膠囊 (Suggestion Chips) - 替代側邊欄按鈕
suggestions = ["🎯 幫我拆解目標", "😫 我現在好焦慮", "🐢 我想動但動不了", "✅ 我完成了！幫我紀錄"]
cols = st.columns(4)
selected_prompt = None

for i, suggestion in enumerate(suggestions):
    if cols[i].button(suggestion):
        selected_prompt = suggestion

# 輸入處理
if prompt := (st.chat_input("告訴我你現在的狀態...") or selected_prompt):
    # 1. 顯示 User Message
    if not selected_prompt: # 如果是按鈕觸發的，上面已經顯示過了，這裡不用重複(Streamlit邏輯)
        pass 
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # 2. 執行 Agent
    with st.spinner("Mind Flow 團隊正在協作中..."):
        result = app.invoke({"messages": st.session_state.messages})
        response = result["messages"][-1]
        
    # 3. 顯示 AI Response
    st.session_state.messages.append(response)
    st.chat_message("assistant").write(response.content)
    
    # 4. 如果有 Tool Call (Architect)，顯示成功提示
    if response.tool_calls:
        st.toast("✨ 日記已寫入資料庫！查看側邊欄數據。", icon="✅")