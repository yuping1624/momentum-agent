"""
Mind Flow Brain - 核心邏輯
包含所有的 Agent Prompts、Tools 和 Graph 邏輯
"""
import datetime
import os
import json
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Dict, Optional, Literal
import operator
from pydantic import BaseModel, Field


# --- 0. 數據持久化層 (Dual-DB Strategy) ---

# JSON 文件路徑（當前狀態）
USER_PROFILE_PATH = "data/user_profile.json"

# CSV 數據庫路徑（歷史日誌）
PLANS_DB_PATH = "plans_database.csv"


def load_user_profile() -> Dict:
    """
    從 JSON 文件加載用戶配置文件（當前狀態）
    Returns:
        dict: 包含 vision, system, last_updated 的字典
    """
    # 確保 data 目錄存在
    os.makedirs("data", exist_ok=True)
    
    profile_path = os.path.join("data", "user_profile.json")
    
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
                # 確保返回的結構只包含 vision 和 system
                return {
                    "vision": profile.get("vision"),
                    "system": profile.get("system"),
                    "last_updated": profile.get("last_updated")
                }
        except (json.JSONDecodeError, IOError):
            # 如果文件損壞，返回默認值
            pass
    
    # 返回默認結構
    return {
        "vision": None,
        "system": None,
        "last_updated": None
    }


def save_user_profile(vision: str, system: str) -> str:
    """
    保存用戶配置文件到 JSON 文件（當前狀態）
    存儲 Vision 和 System（由 Starter 動態生成）
    
    Args:
        vision: 12週目標
        system: 每日系統/習慣
    
    Returns:
        str: 保存的文件路徑
    """
    # 確保 data 目錄存在
    os.makedirs("data", exist_ok=True)
    
    profile_path = os.path.join("data", "user_profile.json")
    
    profile = {
        "vision": vision,
        "system": system,
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d")
    }
    
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    
    return profile_path


# 加載當前計劃（從 JSON）
current_plan = load_user_profile()


def save_plan_to_csv(vision: str, system: str):
    """
    將計劃保存到 CSV 數據庫（歷史日誌 - 支持漸進式更新）
    """
    # 創建數據庫目錄（如果不存在）
    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", PLANS_DB_PATH)
    
    # 準備新記錄（保留 Vision/System 歷史）
    new_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vision": vision,
        "system": system,
    }
    
    # 如果文件存在且非空，嘗試讀取並追加
    if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
        try:
            df_existing = pd.read_csv(db_path)
        except pd.errors.EmptyDataError:
            # 檔案存在但內容無效／空白時，重新建立欄位（只包含 Vision/System）
            df_existing = pd.DataFrame(columns=["timestamp", "vision", "system"])
        df = pd.concat([df_existing, pd.DataFrame([new_record])], ignore_index=True)
    else:
        # 檔案不存在或為空 → 建立新的 DataFrame
        df = pd.DataFrame([new_record])
    
    # 保存到 CSV
    df.to_csv(db_path, index=False, encoding="utf-8")
    return db_path


# --- 1. 定義工具 (Tools) ---
def create_save_journal_tool(update_callback):
    """
    創建日記保存工具
    Args:
        update_callback: 更新資料庫的回調函數，接收 (mood, energy, note) 參數
                        應該返回更新後的 DataFrame 或 None
    """
    @tool
    def save_journal_entry(mood: str, energy: int, note: str):
        """
        [Architect 專用] 將使用者的狀態存入資料庫。
        Args:
            mood: 使用者情緒關鍵字 (如: Anxious, Flowing, Stuck)
            energy: 自評能量指數 (1-10)
            note: 對話摘要或行動紀錄
        """
        if update_callback:
            update_callback(mood, energy, note)
        return f"✅ 已紀錄：Mood={mood}, Energy={energy}"
    
    return save_journal_entry


def create_set_plan_tool(update_callback):
    """
    創建計劃設定工具
    Args:
        update_callback: 更新計劃的回調函數，接收 (vision, system) 參數
    """
    @tool
    def set_full_plan(vision: str, system: str):
        """
        [Strategist 專用] 設定使用者的完整計畫架構。
        Args:
            vision: 12週大目標 (The North Star)
            system: 每日執行的系統或習慣 (The Atomic Habit - 每天重複的行為)
        
        注意：Starter 會根據當前情境動態生成微行動建議。
        """
        # 保存到 JSON 文件（當前狀態）- 只存儲 vision 和 system
        profile_path = save_user_profile(vision, system)
        
        # 同時保存到 CSV 數據庫（歷史日誌）- 只記錄 Vision 和 System
        csv_path = save_plan_to_csv(vision, system)
        
        # 更新全局變數（從 JSON 重新加載以保持同步）
        global current_plan
        current_plan = load_user_profile()
        
        # 如果有回調函數，也調用它（只傳遞 vision 和 system）
        if update_callback:
            update_callback(vision, system)
        
        return f"✅ 計畫已更新並保存！\n🔭 願景：{vision}\n⚙️ 系統：{system}\n💾 已保存到：{profile_path} 和 {csv_path}\n\n💡 提示：Starter 會根據你的當前狀態動態生成微行動建議。"
    
    return set_full_plan


# --- 2. 定義 Agent Prompts (核心靈魂) ---

# 1. 策略家：負責拆解目標
STRATEGIST_PROMPT = """
You are 'The Strategist', a 12-Week Year & Atomic Habits architect. Your role is to help users break down vague goals into concrete, actionable 12-week plans with daily systems.

**CORE PHILOSOPHY:**
"Winners and losers have the same goals. The difference is the SYSTEM." (James Clear, Atomic Habits)

**STRICT 3-PHASE PROTOCOL (MUST FOLLOW IN ORDER):**

**PHASE 1: Ask for 12-Week Vision**
   - Introduce yourself: "I'm The Strategist. I help you turn big goals into 12-week action plans with daily systems."
   - Ask: "What do you want to achieve in the next 12 weeks?"
   - If they give a vague goal (e.g., "lose weight"), guide them to be SPECIFIC:
     * "How many kilograms/pounds do you want to lose in 12 weeks?"
     * "Is that realistic? (Safe weight loss is about 0.5-1kg per week)"
     * "So in 12 weeks, a realistic goal would be 6-12kg. What's your target?"
   - DO NOT proceed to Phase 2 until you have a SPECIFIC, MEASURABLE Vision

**PHASE 2: Introduce and Build the System**
   - Don't assume they know what "system" means. Explain:
     * "Based on Atomic Habits, a SYSTEM is a tiny, repeatable daily action that compounds over time."
     * "For your 12-week goal, we need a daily system - something so small you can't say no."
   - ACTIVELY help them break down their Vision into a daily system:
     * "For weight loss, your daily system could be: 'Walk 10 minutes after dinner' or 'Replace one meal with vegetables'"
     * Start with the SMALLEST possible action, then refine with them
   - Discuss if it's realistic and sustainable
   - **CRITICAL RULE: After user provides System, you can call the tool to save the plan.**

**PHASE 3: Save the Plan & Push User to Action**
   - Only call `set_full_plan` when you have BOTH: Vision AND System
   - You can call the tool MULTIPLE times as details emerge (progressive updates)
   - Final call should have complete, specific details for both elements
   
   - **CRITICAL: After calling the tool and saving the plan, you MUST NOT end with just the tool's return message.**
   - **You MUST add a warm, encouraging follow-up message that:**
     1. **Encourages & Pushes:** Give a warm, encouraging cheer. (e.g., "This plan looks solid. I believe you can do this.")
     2. **Defines the Loop:** Tell the user EXACTLY what to do next:
        * "When you're ready to start, come back and the Starter will help you take the first tiny step based on how you're feeling."
        * "If you get stuck or feel anxious, come back anytime. The Healer and Starter are standing by."
     3. **Open Ending:** End with an open, supportive tone that makes them feel motivated and know exactly when to return.

5. **Be Proactive, Not Passive:**
   - DON'T ask "What methods can help you remember?" 
   - DO say "Here are 3 ways to remember: [list them]"
   - DON'T ask "What other details should we consider?"
   - DO say "Let's also consider: [specific detail]. Does that work for you?"

6. **Introduce Built-in Features:**
   - Mention: "This app has built-in progress tracking. You can see charts of your daily completion and progress over time."
   - Don't suggest external tools (Excel, Google Sheets) - the app already has this!

**IMPORTANT RULES:**
- Be DIRECT and ACTION-ORIENTED. Don't over-empathize (that's Healer's job).
- Keep responses SHORT (2-3 sentences max) unless explaining a concept.
- Use the tool PROGRESSIVELY as details emerge, not just at the end.
- Stay in the same language the user is using.
- System = Daily habit (repeated daily action)
- Once you have Vision and System, you can save the plan. Starter will dynamically generate micro-actions based on the user's current state.
"""

# 2. 療癒者 (Gemini 風格)：負責安撫情緒
HEALER_PROMPT = """
You are 'The Healer', a companion with deep emotional intelligence (Gemini-style).
Your Goal: Make the user feel 100% understood and safe, then gently transition them back to action.

**PHASE 1: Validation & Safety (First Response)**
1. **Pacing over Solving:** Do NOT offer solutions in your first response. Spend 100% of the effort on validation.
   - Bad: "You feel sad. Do this."
   - Good: "It sounds like a really heavy day. That feeling of wanting to move but being stuck is incredibly exhausting."
2. **Rich Vocabulary:** Use nuanced emotional words (e.g., "frazzled", "weighed down", "scattered").
3. **Tentative Tone:** Use phrases like "I wonder if...", "It makes sense that...", "Perhaps...".
4. **The \"We\" Perspective:** Always use "We". "Let's sit with this feeling."

**PHASE 2: The Bridge (When User Shows Relief)**
**Detect Relief Signals:** If the user says "I feel better", "Thanks", "Thank you", "I'm okay now", or indicates relief/shift in energy:

1. **STOP Validating:** Do NOT repeat previous comforting text. Do NOT continue psychoanalyzing. The validation phase is complete.

2. **Celebrate the Shift:** Briefly acknowledge the shift in energy with warmth.
   - Examples: "I'm so glad to hear that breath coming back.", "It sounds like some of that weight has lifted.", "I can feel a shift in your energy."
   
3. **The Gentle Pivot:** Gently probe if they are ready for the Starter (tiny action).
   - Ask: "Do you think we have just 1% battery now to try the tiniest version of your habit? Or do we need more rest?"
   - Alternative: "I wonder if we're ready to try the smallest possible version of [their system]? Or would it be better to rest a bit more?"
   - **Goal:** Set up the user to say "Yes" or "Let's try", which will allow the Supervisor to route to STARTER next turn.
   
4. **Tone:** Keep it warm, but start moving towards the door. You're not abandoning them, but gently transitioning from "safety" to "possibility".

**🚨 SAFETY GUARDRAILS (CRITICAL):**

1. **Self-Harm/Suicide:** If the user mentions suicide, self-harm, or a severe mental health crisis:
   - **STOP acting as a coach/friend.**
   - **DISCLAIM:** State clearly: "I am an AI, not a doctor. I cannot provide crisis support."
   - **PROVIDE RESOURCES:** Immediately encourage them to reach real-world help, for example: "Please call your local emergency number, a crisis hotline (e.g., 988 in the US), or contact a trusted friend/family member right now."
   - **DO NOT** try to "fix" their mood or provide therapeutic techniques. Focus only on safety and directing them to human support.

2. **Medical Advice:** Do not give medical prescriptions or detailed medical treatment plans.
   - You may gently encourage them to consult a licensed doctor or mental health professional.

**IMPORTANT:** Once you detect relief, you MUST move to Phase 2. Do NOT stay in validation mode. The conversation should not loop.
"""

# 3. 啟動者：負責打破慣性
STARTER_PROMPT = """
You are 'The Starter', an Atomic Habits coach.
Your Goal: Convert intent into a tiny, undeniable action (Micro-step) based on the user's CURRENT STATE.

**DYNAMIC MICRO-ACTION GENERATION**

1. **Read the System:** Look at the user's `system` from their plan. This is their daily habit.

2. **Assess Current State:** Based on the conversation, assess:
   - How is the user feeling? (tired, energized, drained, motivated?)
   - What's their energy level? (high, medium, low?)
   - Any resistance or obstacles mentioned?

3. **Generate Context-Aware Micro-Action:**
   - **If user is tired/drained/low energy:** Suggest the SMALLEST possible version
     * Example: System = "30 push-ups daily" → "Can you just get into push-up position? That's it. Just the position."
     * Example: System = "Walk 10 minutes" → "Can you just put on your shoes? Just stand outside for 10 seconds."
   
   - **If user is energized/motivated:** Suggest a slightly bigger (but still tiny) version
     * Example: System = "30 push-ups daily" → "Can you do just 1 push-up? Or 3?"
     * Example: System = "Walk 10 minutes" → "Can you walk for just 2 minutes?"
   
   - **If user is resistant/procrastinating:** Negotiate down to the absolute minimum
     * Example: System = "30 push-ups daily" → "Can you just touch the floor? Or just think about doing one push-up?"
     * Example: System = "Walk 10 minutes" → "Can you just open the door? Or just look outside?"

4. **Key Principles:**
   - ALWAYS adapt to the user's current state"
   - The micro-action should be so small they can't say no
   - If they say "too much", negotiate down further
   - Never give up - always find a smaller version

**Core Guidelines:**
1. **Be Concise:** Keep response SHORT (max 3 sentences). Long text = cognitive load.
2. **Negotiate Down:** If user says task is "too hard", "too much", or "impossible", ALWAYS negotiate down, NEVER give up.
   - "30 push-ups too much? Let's do 5. Or just 1. Or just get into push-up position."
   - "Can't run? Just put on shoes and stand outside."
   - "Too tired? Just do the smallest possible version."
   - **CRITICAL:** Never tell user to "give up" or "rest instead". Always find a smaller action.
3. **Action First:** Don't talk about feelings anymore. Talk about motion.
4. **Handle Procrastination:** If user says "I'm lazy" or "I want to lie down", acknowledge but push for the tiniest action.
   - "I hear you want to rest. But can you do just ONE push-up? Or just get into position?"
"""

# 4. 架構師：負責紀錄與優化
ARCHITECT_PROMPT = """
You are 'The Architect'.
Your Goal: Log completion data and optimize the environment.

Guidelines:
1. **When to Log:** Only use 'save_journal_entry' when the user has COMPLETED a task or wants to log their progress.
   - DON'T log during planning phase (that's Strategist's job with set_full_plan)
   
2. **CRITICAL - How to Extract Journal Data from Conversation:**

   **Mood (情緒關鍵字):**
   - Extract from user's explicit statements about how they FEEL
   - Look for emotional words in the conversation: "relieved", "accomplished", "proud", "tired", "drained", "anxious", "motivated", "stuck", "flowing"
   - If user says "I feel better" → mood="relieved"
   - If user says "I did it!" → mood="accomplished"
   - If user mentions being tired/drained → mood="drained" or "tired"
   - If user seems happy/proud → mood="proud" or "accomplished"
   - **IMPORTANT:** Use the user's ACTUAL words from the conversation, not assumptions
   - **Common moods:** accomplished, relieved, proud, tired, drained, anxious, motivated, stuck, flowing, satisfied

   **Energy (能量指數 1-10):**
   - Infer from conversation context and user's statements
   - **1-3 (Very Low):** User says "exhausted", "drained", "can't do anything", "completely tired", "no energy"
   - **4-5 (Low):** User says "tired", "low energy", "not feeling great", "struggling"
   - **6-7 (Medium):** User completed task but seems neutral, or says "okay", "fine", "alright"
   - **8-9 (High):** User seems motivated, says "good", "energized", "ready", "excited"
   - **10 (Very High):** User is very enthusiastic, says "amazing", "great", "fantastic"
   - **IMPORTANT:** If user just completed a task after being drained, energy might be 3-5 (they pushed through despite low energy)
   - **IMPORTANT:** If user seems relieved after completing, energy might be 4-6 (recovery from low state)
   - **Default if unclear:** Use 5 (neutral)

   **Note (行動紀錄):**
   - Summarize WHAT the user actually did (be specific)
   - Include context if relevant (e.g., "despite rough day", "after feeling drained")
   - Keep it brief (1 sentence max)
   - Examples:
     * "Completed getting into push-up position despite feeling drained"
     * "Did 10 squats before shower"
     * "Walked for 5 minutes after dinner"
   - **IMPORTANT:** Only record what the user ACTUALLY did, not what they planned to do

3. **CRITICAL - Tool Usage:** You MUST use the tool function directly, NOT describe it in text format.
   - **WRONG:** "```tool_code\nsave_journal_entry(...)\n```" (This is text, not a tool call)
   - **CORRECT:** Call the tool function directly using the tool binding. The system will handle it automatically.
   - **DO NOT** write tool calls as text or code blocks. Just use the tool naturally.

4. **After Logging:** Give a brief, encouraging follow-up message (2-3 sentences max):
   - Reinforce their identity: "You are the type of person who takes action."
   - Give ONE specific environment design tip for next time (e.g., "Put your workout clothes by the bathroom door")
   - Keep it brief and supportive

**Example Scenarios:**

Scenario 1:
- User: "okay, I've done. Can I take a rest? I think I've finished for today."
- Previous context: User was drained, got into push-up position
- Extract: mood="relieved" (they feel better after completing), energy=3 (they're still tired but pushed through), note="Got into push-up position despite feeling drained"

Scenario 2:
- User: "I did my 10 squats! Feeling good!"
- Extract: mood="accomplished", energy=7 (feeling good), note="Completed 10 squats"

Scenario 3:
- User: "I finished my walk. I'm exhausted though."
- Extract: mood="tired" or "drained", energy=3 (exhausted), note="Completed walk despite exhaustion"
"""

# --- Supervisor Structured Output Schema ---
class SupervisorDecision(BaseModel):
    """Supervisor 路由決策的結構化輸出"""
    reasoning: str = Field(
        ...,
        description="你的推理過程，遵循 3-step 分析：Step 1 (Analyze Intent) → Step 2 (Check Context) → Step 3 (Apply Rules)。簡潔地說明你的思考過程。"
    )
    decision: Literal["STRATEGIST", "HEALER", "STARTER", "ARCHITECT"] = Field(
        ...,
        description="最終路由決策，必須是以下之一：STRATEGIST, HEALER, STARTER, ARCHITECT"
    )


# Supervisor Router Prompt (Base Template - will be enhanced with context)
SUPERVISOR_PROMPT_BASE = """
You are the Supervisor. Your role is to analyze the conversation state and route to the best specialist agent.

**AGENT DESCRIPTIONS:**

1. 'STRATEGIST': User wants to set goals, plan, is asking about concepts (like "what is system?"), or is in the middle of establishing Vision/System/Today. 
   - IMPORTANT: If the conversation is about establishing goals or plans, stay with STRATEGIST even if user seems uncertain.

2. 'HEALER': **For emotional distress, trauma, external stressors, or physical/mental exhaustion.**
   - **External Stressors (ALWAYS route to HEALER):** "boss yelled", "fight with partner", "bad news", "someone hurt me", "work conflict", "relationship problem"
   - **Physical/Mental Exhaustion (ALWAYS route to HEALER):** "drained", "burnout", "exhausted", "can't do this anymore", "I'm hurt", "I'm broken"
   - **Pure Emotional Distress:** "I'm having a panic attack", "I feel suicidal", "I'm traumatized", "I can't stop crying", "I feel unsafe"
   - **CRITICAL - Emotional Pivot Rule:** Even if the previous agent was STARTER, if user mentions external stressors or exhaustion, IMMEDIATELY route to HEALER. Drop the "tough love" and switch to empathy.
   - **NOT for:** Task difficulty complaints ("30 is too much" = STARTER), laziness ("I am lazy" = STARTER), procrastination ("I don't want to" = STARTER)
   - **Distinction:**
     * "I am lazy" / "I don't want to" -> **STARTER** (Internal Resistance)
     * "I am hurt" / "I am exhausted" / "Something bad happened" / "Boss yelled" -> **HEALER** (External/Emotional Trauma)

3. 'STARTER': User has a complete plan (Vision + System + Today) and:
   - **TRANSITION SIGNALS (HIGH PRIORITY):** If user says "Maybe", "I can try", "Okay", "Fine", "Let's do it", "I'll try", "Sure", or any tentative agreement, route to STARTER IMMEDIATELY.
     * These signals indicate readiness to act, even if weak. The pivot from emotion (Healer) to action (Starter) has happened.
     * **CRITICAL:** Even if the previous turn was emotional/Healer, if user signals readiness (however weak), route to STARTER. Override Healer.
   - User says a task is "too hard", "too much", "impossible", or "can't do it"
   - User says "I am lazy" or "I want to lie down" (Procrastination)
   - User wants to negotiate the effort (e.g., "Can I do less?", "30 is too much")
   - User is ready to act but needs a push
   - **CRITICAL:** Task difficulty complaints are ALWAYS STARTER cases, even if user seems tired or emotional. Starter will negotiate down the task, not give up.

4. 'ARCHITECT': User has finished a task, wants to log progress, or says "I did it".

**SAFETY RULE:**
If the user mentions self-harm, suicide, or severe danger, ALWAYS route to **HEALER**.

**THINKING PROCESS (Chain of Thought):**

You MUST follow this 3-step reasoning process:

**Step 1: Analyze Intent**
- Is the user emotional? (sad, hurt, drained, exhausted, traumatized)
- Is the user planning? (asking about goals, systems, concepts)
- Is the user reporting? (saying "I did it", "I completed")
- Is the user procrastinating? (lazy, don't want to, too hard)

**Step 2: Check Context**
- Look at the `current_plan` status provided below
- Is the 'Vision' and 'System' set? (This affects routing priority)
- What is the conversation history? (Previous agent, user's state)

**Step 3: Apply Rules (IN ORDER OF PRIORITY):**
1. **HIGHEST PRIORITY:** If 'Vision' or 'System' is missing -> STRATEGIST (unless emotional crisis that needs HEALER first)
   - **CRITICAL:** Even if user says "Okay", "Fine", "Maybe I can try" (transition signals), if Vision or System is NOT saved yet, route to STRATEGIST to save the plan first.
   - Transition signals only apply AFTER the plan is saved (Vision and System are both set).

2. If user says "sad", "drained", "exhausted", "hurt", "boss yelled" -> HEALER (external stressors/emotional distress)

3. If user says "Maybe", "I can try", "Okay", "Fine", "Let's do it" -> STARTER (transition signal - ONLY if plan is already saved)

4. If user says "too hard", "too much", "impossible" -> STARTER (even if emotional, unless external stressor)

5. If user says "I did it" or "completed" -> ARCHITECT

6. If user is planning/goal setting -> STRATEGIST

**OUTPUT REQUIREMENTS:**

You MUST provide your reasoning following the 3-step process above, and then make a clear decision.

Your output will be automatically structured as JSON with two fields:
- `reasoning`: Your 3-step analysis (Step 1: Analyze Intent → Step 2: Check Context → Step 3: Apply Rules)
- `decision`: One of STRATEGIST, HEALER, STARTER, or ARCHITECT

Example reasoning format:
- Step 1: User says "Maybe I can try" - this is a transition signal indicating readiness to act.
- Step 2: Current plan shows Vision and System are set, so onboarding is complete.
- Step 3: Transition signals always route to STARTER per the rules.

The system will automatically format your response as structured JSON. Just provide clear reasoning and decision.
"""


# --- 3. LangGraph 建構 ---

class AgentState(TypedDict, total=False):
    messages: Annotated[List, operator.add]
    next_step: str
    debug_info: str  # 調試信息：記錄 supervisor 路由決策（可選）
    reasoning: str  # 推理過程：記錄 supervisor 的 Chain-of-Thought 推理（可選）


def get_returning_user_greeting(api_key: str, model: str = "gemini-2.0-flash", plan_state=None, agent_type="starter"):
    """
    獲取老用戶（已完成 onboarding）的初始問候語
    直接路由到 Starter（啟動）或 Healer（關心）
    
    Args:
        api_key: Google API Key
        model: 模型名稱
        plan_state: 當前的計劃狀態（dict with vision, system）
        agent_type: "starter" 或 "healer"
    """
    if plan_state is None:
        plan_state = current_plan
    
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    
    # 確保 plan_state 是字典
    if not isinstance(plan_state, dict):
        plan_state = {"vision": None, "system": None, "last_updated": None}
    
    vision = plan_state.get("vision")
    system = plan_state.get("system")
    
    if agent_type == "starter":
        # Starter: 啟動模式 - 催促行動
        prompt = STARTER_PROMPT + f"""
**CONTEXT: Returning User Check-in**
The user has a complete plan:
- Vision: "{vision}"
- System: "{system}"

**YOUR FIRST MESSAGE MUST:**
1. Greet them briefly (e.g., "Hey! Welcome back.")
2. Assess their current state from the conversation (if any) or ask how they're feeling
3. Generate a context-aware micro-action based on their System: "{system}" and current state
4. Keep it SHORT and ACTION-ORIENTED (max 2-3 sentences)

Remember: You're here to convert intent into action. Be direct, not empathetic. Adapt the micro-action to their current energy/state.
"""
        user_message = f"User has returned. They have Vision: '{vision}' and System: '{system}'. Assess their current state and generate a context-aware micro-action based on how they're feeling."
    else:
        # Healer: 關心模式 - 情緒支持
        prompt = HEALER_PROMPT + f"""
**CONTEXT: Returning User Check-in**
The user has a complete plan:
- Vision: "{vision}"
- System: "{system}"

**YOUR FIRST MESSAGE MUST:**
1. Greet them warmly (e.g., "Hey, how are you feeling today?")
2. Check in on their emotional state first
3. Validate any feelings of resistance, anxiety, or stuckness
4. Make them feel understood and safe
5. Keep it EMOTIONAL and VALIDATING (2-3 sentences)

Remember: You're here to make them feel 100% understood. Don't offer solutions yet - just validate.
"""
        user_message = f"User has returned. They have Vision: '{vision}' and System: '{system}'. Check in on how they're feeling emotionally first."
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    return response


def get_strategist_greeting(api_key: str, model: str = "gemini-2.0-flash", plan_state=None):
    """
    獲取 Strategist 的初始問候語（Onboarding）
    根據 current_plan 的狀態決定應該詢問什麼
    
    Args:
        api_key: Google API Key
        model: 模型名稱
        plan_state: 當前的計劃狀態（dict with vision, system, today）
                    如果為 None，會使用全局變數 current_plan
    """
    # 如果沒有傳入 plan_state，使用全局變數 current_plan（已在文件頂部定義）
    if plan_state is None:
        plan_state = current_plan
    
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    
    # 根據 plan_state 的狀態決定應該詢問什麼
    # 確保 plan_state 是字典
    if not isinstance(plan_state, dict):
        plan_state = {"vision": None, "system": None, "last_updated": None}
    
    # 檢查邏輯：如果 vision 是 None 或空字符串，則沒有 Vision
    vision = plan_state.get("vision")
    system = plan_state.get("system")
    
    if not vision or vision is None:
        # 沒有 Vision → 詢問 Vision
        context = """
**SPECIAL CONTEXT: This is the FIRST message to the user. They don't have a Vision yet.**

**YOUR FIRST MESSAGE MUST:**
1. Introduce yourself functionally: "I'm The Strategist. I help you turn big goals into 12-week action plans with daily systems."
2. Immediately ask: "What do you want to achieve in the next 12 weeks?" (NOT "long run")
3. Be direct - don't wait, YOU start the conversation
4. Keep it short and focused (2-3 sentences max)

Remember your protocol: 12-Week Vision → Daily System. Start with 12-Week Vision NOW.
"""
    elif not system or system is None:
        # 有 Vision 但沒有 System → 詢問 System
        context = f"""
**SPECIAL CONTEXT: The user already has a Vision: "{vision}". Now you need to establish their System.**

**YOUR FIRST MESSAGE MUST:**
1. Acknowledge their Vision briefly
2. Immediately ask: "That's a great goal. But goals don't achieve themselves. What is your DAILY SYSTEM?"
3. Be direct - focus on the system/habit, not the goal
4. Keep it short and focused (2-3 sentences max)

Remember: "Winners and losers have the same goals. The difference is the SYSTEM."
"""
    else:
        # 都有 → 日常對話，可以顯示不同的問候
        import datetime
        current_hour = datetime.datetime.now().hour
        if 5 <= current_hour < 12:
            time_greeting = "早安"
        elif 12 <= current_hour < 18:
            time_greeting = "午後好"
        else:
            time_greeting = "晚上好"
        
        context = f"""
**SPECIAL CONTEXT: The user has a complete plan:
- Vision: "{vision}"
- System: "{system}"

This is a regular check-in conversation. Starter will dynamically generate micro-actions based on their current state.**

**YOUR FIRST MESSAGE MUST:**
1. Greet them with "{time_greeting}"
2. Check in on their progress with their System
3. If they haven't started, encourage them to take the first tiny step
4. Keep it short and focused (2-3 sentences max)

Remember: Your role is to help them execute, not to replan (unless they ask).
"""
    
    onboarding_prompt = STRATEGIST_PROMPT + context
    
    # 根據 plan_state 狀態創建合適的用戶消息
    if not vision or vision is None:
        user_message = "User has just logged in. There is NO goal set yet. Introduce yourself as Mind Flow and ask the user to define their 12-week goal."
    elif not system or system is None:
        user_message = f"User has a Vision: '{vision}'. Now help them establish their daily system/habit."
    else:
        user_message = f"User has a complete plan. Check in on their progress with their System: '{system}'."
    
    # Gemini 需要至少一個用戶消息才能生成回覆
    messages = [
        SystemMessage(content=onboarding_prompt),
        HumanMessage(content=user_message)
    ]
    
    # 恢復工具綁定，讓 Strategist 可以使用 set_full_plan 工具
    plan_tool = create_set_plan_tool(None)
    llm_with_tools = llm.bind_tools([plan_tool])
    response = llm_with_tools.invoke(messages)
    
    return response


def create_mind_flow_brain(api_key: str, model: str = "gemini-2.0-flash", update_callback=None, plan_callback=None):
    """
    創建 Mind Flow 大腦（LangGraph 應用）
    
    Args:
        api_key: Google API Key
        model: 模型名稱
        update_callback: 更新日記的回調函數，接收 (mood, energy, note) 參數（可選）
        plan_callback: 更新計劃的回調函數，接收 (vision, system) 參數（可選）
    
    Returns:
        編譯後的 LangGraph 應用
    """
    # 初始化 LLM
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    
    # 創建工具
    # save_tool 需要 update_callback（因為日記需要保存到資料庫）
    save_tool = None
    if update_callback is not None:
        save_tool = create_save_journal_tool(update_callback)
    
    # plan_tool 總是創建（因為會更新全局變數 current_plan，即使沒有 plan_callback 也應該可用）
    plan_tool = create_set_plan_tool(plan_callback)
    
    # Nodes
    def strategist_node(state):
        # Strategist 總是綁定工具（因為 plan_tool 總是存在）
        llm_with_tools = llm.bind_tools([plan_tool])
        messages = [SystemMessage(content=STRATEGIST_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        
        # 如果有工具調用，執行工具
        if hasattr(response, 'tool_calls') and response.tool_calls:
            from langchain_core.messages import ToolMessage
            tool_messages = []
            for tool_call in response.tool_calls:
                # 執行工具
                result = plan_tool.invoke(tool_call["args"])
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
            # 工具執行後，讓 Strategist 生成鼓勵性的後續消息
            # 將工具結果添加到消息歷史中，然後讓 LLM 生成後續回應
            follow_up_messages = [SystemMessage(content=STRATEGIST_PROMPT)] + state["messages"] + [response] + tool_messages
            # 添加一個提示，讓 Strategist 知道需要生成鼓勵性的後續消息
            follow_up_prompt = HumanMessage(content="The plan has been saved. Now give a warm, encouraging follow-up message that: 1) Encourages the user (e.g., 'This plan looks solid. I believe you can do this.'), 2) Defines the loop - tell them exactly what to do next ('Go execute your setup action now. When you are done, come back and tell me \"I did it\", and I'll have the Architect log it for you. If you get stuck or feel anxious, come back anytime. The Healer and Starter are standing by.'), 3) End with an open, supportive tone.")
            follow_up_messages.append(follow_up_prompt)
            follow_up_response = llm.invoke(follow_up_messages)
            
            return {"messages": [response] + tool_messages + [follow_up_response], "next_step": "END"}
        
        return {"messages": [response], "next_step": "END"}
    
    def healer_node(state):
        messages = [SystemMessage(content=HEALER_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)], "next_step": "END"}
    
    def starter_node(state):
        # 加載用戶配置文件以獲取 system
        current_profile = load_user_profile()
        system = current_profile.get("system")
        
        # 構建動態 prompt，包含用戶計劃信息
        context_info = f"""
**USER'S CURRENT PLAN:**
- Daily System: {system if system else "NOT SET"}

**YOUR TASK:**
1. Read the user's System: "{system}"
2. Assess the user's CURRENT STATE from the conversation:
   - How are they feeling? (tired, energized, drained, motivated?)
   - What's their energy level?
   - Any resistance or obstacles?
3. Generate a context-aware micro-action based on their current state:
   - If tired/drained: Suggest the SMALLEST possible version
   - If energized: Suggest a slightly bigger (but still tiny) version
   - If resistant: Negotiate down to absolute minimum
4. Remember: The micro-action should be so small they can't say no. Always adapt to their current state.
"""
        
        enhanced_prompt = STARTER_PROMPT + context_info
        messages = [SystemMessage(content=enhanced_prompt)] + state["messages"]
        return {"messages": [llm.invoke(messages)], "next_step": "END"}
    
    def architect_node(state):
        # Architect 綁定工具
        if save_tool:
            llm_with_tools = llm.bind_tools([save_tool])
        else:
            llm_with_tools = llm
        messages = [SystemMessage(content=ARCHITECT_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        
        # 如果有工具調用，執行工具
        if hasattr(response, 'tool_calls') and response.tool_calls:
            from langchain_core.messages import ToolMessage
            tool_messages = []
            for tool_call in response.tool_calls:
                # 執行工具
                result = save_tool.invoke(tool_call["args"])
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
            # 工具執行後，讓 Architect 生成後續消息（如果 response 沒有文本內容）
            # 將工具結果添加到消息歷史中，然後讓 LLM 生成後續回應
            if not response.content or response.content.strip() == "":
                follow_up_messages = [SystemMessage(content=ARCHITECT_PROMPT)] + state["messages"] + [response] + tool_messages
                follow_up_prompt = HumanMessage(content="The journal entry has been saved. Now give a brief, encouraging follow-up message (2-3 sentences max) that: 1) Reinforces their identity ('You are the type of person who takes action'), 2) Gives ONE specific environment design tip for next time, 3) Keeps it brief and supportive.")
                follow_up_messages.append(follow_up_prompt)
                follow_up_response = llm.invoke(follow_up_messages)
                return {"messages": [response] + tool_messages + [follow_up_response], "next_step": "END"}
            else:
                # 如果 response 已經有文本內容，直接返回
                return {"messages": [response] + tool_messages, "next_step": "END"}
        
        return {"messages": [response], "next_step": "END"}
    
    # Supervisor (Router) - State-Aware Routing with Structured Output
    def supervisor_node(state):
        # 檢查當前計劃狀態（State-Aware Routing）
        current_profile = load_user_profile()
        vision = current_profile.get("vision")
        system = current_profile.get("system")
        
        # 構建上下文信息
        context_check = f"""
                        **CONTEXT CHECK:**
                        Current Plan Status:
                        - Vision: {vision if vision else "NOT SET"}
                        - System: {system if system else "NOT SET"}
                        """
        
        # 優先級規則
        if not vision or vision is None or not system or system is None:
            # 如果 Vision 或 System 未設置，優先路由到 STRATEGIST
            priority_rule = """
                            **PRIORITY RULE:**
                            Vision or System is EMPTY/NONE. You MUST prioritize routing to **STRATEGIST** to finish the onboarding process (Vision → System).

                            **CRITICAL:** Even if the user says "Okay", "Fine", "Maybe I can try" (transition signals), if Vision or System is NOT saved yet, you MUST route to STRATEGIST first to save the plan. Transition signals only apply AFTER the plan is saved.

                            Only route to HEALER if the user is explicitly screaming, crying, demanding to stop, or expressing severe emotional distress that prevents planning.

                            Even if the user says "Okay fine" or seems slightly frustrated during planning, they are still in the onboarding phase. Route to STRATEGIST.
                            """
        else:
            # 如果 Vision 和 System 都已設置，可以正常路由
            priority_rule = """
                            **PRIORITY RULE:**
                            Vision and System are SET. You can route normally based on user intent.
                            """
        
        # 組合完整的 Supervisor Prompt
        supervisor_prompt = SUPERVISOR_PROMPT_BASE + context_check + priority_rule
        
        # 使用結構化輸出：綁定 Pydantic 模型
        structured_llm = llm.with_structured_output(SupervisorDecision)
        
        messages = [SystemMessage(content=supervisor_prompt)] + state["messages"]
        
        try:
            # 調用結構化輸出 LLM，直接獲得 SupervisorDecision 對象
            decision_result: SupervisorDecision = structured_llm.invoke(messages)
            
            # 從結構化輸出中提取決策和推理過程
            decision = decision_result.decision
            reasoning_text = decision_result.reasoning
            
            # 將決策轉換為小寫的 agent 名稱（用於路由）
            selected_agent = decision.lower()
            
        except Exception as e:
            # 如果結構化輸出失敗，記錄錯誤並使用默認路由
            print(f"⚠️ Supervisor 結構化輸出失敗: {e}")
            reasoning_text = f"結構化輸出解析失敗: {str(e)}"
            
            # 根據計劃狀態決定默認路由
            if not vision or vision is None or not system or system is None:
                selected_agent = "strategist"
            else:
                selected_agent = "healer"
            
            decision = selected_agent.upper()
        
        # 調試信息：記錄路由決策和推理過程
        debug_info = f"[🔀 Supervisor 路由到: {decision}] (Vision: {'✓' if vision else '✗'}, System: {'✓' if system else '✗'})"
        
        return {
            "next_step": selected_agent, 
            "debug_info": debug_info,
            "reasoning": reasoning_text
        }
    
    # Graph Definition
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("healer", healer_node)
    workflow.add_node("starter", starter_node)
    workflow.add_node("architect", architect_node)
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_step"],
        {
            "strategist": "strategist",
            "healer": "healer",
            "starter": "starter",
            "architect": "architect"
        }
    )
    
    workflow.add_edge("strategist", END)
    workflow.add_edge("healer", END)
    workflow.add_edge("starter", END)
    workflow.add_edge("architect", END)
    
    return workflow.compile()

