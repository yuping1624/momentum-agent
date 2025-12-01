"""
Momentum App - Streamlit Interface
Only handles display and user interaction, core logic is in brain.py
"""
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import pandas as pd
import datetime
import os
import html
import re
import altair as alt
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from brain import create_mind_flow_brain, load_user_profile

# --- Safety Keywords (Guardrails) ---
SAFETY_KEYWORDS = [
    # English
    "suicide",
    "kill myself",
    "want to die",
    "want to end it all",
    "end my life",
    "self-harm",
    "self harm",
    # Chinese (kept for detection)
    "自殺",
    "想死",
    "不想活了",
    "活不下去",
    "想結束一切",
    "傷害自己",
]

SAFETY_MESSAGE = (
    "⚠️ I noticed you mentioned content that may be related to self-harm or life safety.\n\n"
    "I am an AI and do not have medical or psychological professional qualifications, "
    "and I cannot provide immediate assistance in emergency situations.\n\n"
    "👉 If you are in **immediate danger**, please contact your local emergency number (e.g., 911) immediately,\n"
    "or call your local suicide prevention/mental health hotline, and seek support from family, friends, or trusted people.\n\n"
    "You deserve to be treated well and to be truly seen and helped."
)

# --- 🛡️ I/O Guardrails ---


def input_guard(user_text: str) -> tuple[bool, str]:
    """
    Input Guardrail: Validates user input before processing.

    Returns:
        tuple: (is_valid, error_message)
        - is_valid: True if input passes all checks
        - error_message: Empty string if valid, otherwise contains error description
    """
    if not isinstance(user_text, str):
        return False, "Invalid input type."

    # Check minimum length
    if len(user_text.strip()) < 2:
        return False, "Please say more so I can help you better."

    # Check maximum length (prevent extremely long inputs that could cause issues)
    if len(user_text) > 10000:
        return False, "Your message is too long. Please break it into smaller parts."

    # Detect prompt injection attempts (basic protection)
    dangerous_patterns = [
        r"ignore\s+all\s+(previous\s+)?instructions?",
        r"forget\s+(all\s+)?(previous\s+)?(rules?|instructions?)",
        r"system\s+prompt",
        r"you\s+are\s+now",
        r"act\s+as\s+if",
        r"pretend\s+to\s+be",
        r"roleplay\s+as",
    ]

    user_lower = user_text.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, user_lower, re.IGNORECASE):
            return False, "I cannot process this type of request. Please rephrase your message."

    return True, ""


def output_guard(ai_text: str) -> str:
    """
    Output Guardrail: Cleans AI response before displaying.

    Removes:
    - tool_code blocks (```tool_code ... ```)
    - Malformed tool call descriptions
    - Other unwanted artifacts

    Returns:
        str: Cleaned text ready for display
    """
    if not isinstance(ai_text, str):
        return str(ai_text) if ai_text else ""

    cleaned = ai_text

    # Remove tool_code blocks (Architect might output these incorrectly)
    # Pattern: ```tool_code ... ``` or ```tool_code\n...\n```
    cleaned = re.sub(
        r'```\s*tool_code\s*.*?```',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove any remaining code blocks that look like tool calls
    # Pattern: ```\n...tool_call...\n``` or similar
    cleaned = re.sub(
        r'```[^\n]*\n.*?(?:save_journal_entry|set_full_plan|tool_call).*?\n```',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove standalone function call patterns that shouldn't be displayed
    # Pattern: save_journal_entry(...) or set_full_plan(...) as plain text
    cleaned = re.sub(
        r'(?:save_journal_entry|set_full_plan)\s*\([^)]*\)',
        '',
        cleaned,
        flags=re.IGNORECASE
    )

    # Clean up excessive whitespace left after removals
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
    cleaned = cleaned.strip()

    return cleaned

# --- RLHF Feedback Logging Function ---


def log_feedback(user_input: str, agent_response: str, rating: int):
    """
    Log user feedback to CSV file.
    rating: 1 = 👍, -1 = 👎
    """
    os.makedirs("data", exist_ok=True)
    feedback_path = os.path.join("data", "feedback_ratings.csv")
    # Clean text to avoid multi-line issues in CSV; convert newlines to readable "\n"

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

# --- Shared Message / Debug Rendering Functions ---


def render_message(msg):
    """Display User / Agent messages on left and right sides respectively, with color blocks."""
    if isinstance(msg, HumanMessage):
        # User on left (blue bubble)
        # Apply output guardrail to user messages too (for consistency and safety)
        content = output_guard(msg.content) if msg.content else ""
        left, right = st.columns([3, 1])
        with left:
            st.markdown(
                f"""
                <div class="mf-msg mf-user">
                    <span class="mf-avatar">👤</span>
                    <span class="mf-text">{html.escape(content)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif isinstance(msg, AIMessage):
        # Agent on right (green bubble)
        # Apply output guardrail to clean any artifacts
        content = output_guard(msg.content) if msg.content else ""
        left, right = st.columns([1, 3])
        with right:
            st.markdown(
                f"""
                <div class="mf-msg mf-agent">
                    <span class="mf-avatar">🤖</span>
                    <span class="mf-text">{html.escape(content)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_supervisor_cot(result):
    """Display Supervisor's reasoning process (Chain of Thought) and routing result on screen (shown above Agent response)."""
    if not isinstance(result, dict):
        return
    reasoning = result.get("reasoning")
    debug_info = result.get("debug_info")
    if not reasoning and not debug_info:
        return

    # Escape HTML and display newlines as <br>, ensuring Step 1/2/3 are clearly separated
    reasoning_html = ""
    if reasoning:
        escaped = html.escape(reasoning)
        reasoning_html = escaped.replace("\r\n", "\n").replace("\n", "<br>")

    debug_html = ""
    if debug_info:
        debug_html = html.escape(debug_info)

    # Display a full-width gray reasoning card
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


# --- Data Persistence and Reading Functions ---
MIND_FLOW_DB_PATH = os.path.join("data", "mind_flow_db.csv")


def load_mind_flow_db():
    """Load journal database from CSV file"""
    os.makedirs("data", exist_ok=True)
    db_path = MIND_FLOW_DB_PATH
    if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
        try:
            df = pd.read_csv(db_path)
            # Ensure required columns exist
            required_cols = ["Timestamp", "Mood", "Energy", "Note"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
            # Remove 'type' column if it exists (for backward compatibility)
            if "type" in df.columns:
                df = df.drop(columns=["type"])
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note"])
    return pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note"])


def save_to_mind_flow_db(timestamp: str, mood: str, energy: int, note: str):
    """Save journal entry to CSV file (with error handling and retry mechanism)"""
    os.makedirs("data", exist_ok=True)
    db_path = MIND_FLOW_DB_PATH

    new_entry = {
        "Timestamp": timestamp,
        "Mood": mood,
        "Energy": energy,
        "Note": note
    }

    try:
        if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
            try:
                df_existing = pd.read_csv(db_path)
                # Remove 'type' column if it exists (for backward compatibility)
                if "type" in df_existing.columns:
                    df_existing = df_existing.drop(columns=["type"])
                # Check if duplicate record exists (avoid duplicates)
                if not df_existing.empty:
                    duplicate = (
                        (df_existing["Timestamp"] == timestamp) &
                        (df_existing["Mood"] == mood) &
                        (df_existing["Energy"] == energy) &
                        (df_existing.get("Note", "") == note)
                    ).any()
                    if duplicate:
                        return df_existing  # Already exists, don't save duplicate
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                df_existing = pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note"])
            df = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([new_entry])

        # Save to CSV, ensure encoding is correct
        df.to_csv(db_path, index=False, encoding="utf-8")
        return df
    except Exception as e:
        # If save fails, log error but don't interrupt program
        print(f"⚠️ Failed to save journal to CSV: {e}")
        # Try to create a backup file
        try:
            backup_path = db_path.replace(".csv", "_backup.csv")
            df.to_csv(backup_path, index=False, encoding="utf-8")
            print(f"✅ Saved to backup file: {backup_path}")
        except Exception:
            pass
        return None


def calculate_dashboard_metrics():
    """Calculate dashboard metrics"""
    journal_db = st.session_state.get("journal_db", pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note"]))

    if journal_db.empty:
        return {
            "total_actions": 0,
            "avg_energy": 0.0,
            "current_streak": 0
        }

    # Total Actions: Total number of journal entries
    total_actions = len(journal_db)

    # Avg Energy: Average of Energy column (handle missing values)
    if "Energy" in journal_db.columns:
        energy_values = pd.to_numeric(journal_db["Energy"], errors="coerce")
        avg_energy = energy_values.mean()
        avg_energy = round(avg_energy, 1) if not pd.isna(avg_energy) else 0.0
    else:
        avg_energy = 0.0

    # Current Streak: Calculate consecutive days with journal entries
    # Count consecutive days from today backwards, see how many consecutive days have records
    if "Timestamp" in journal_db.columns:
        try:
            journal_db_copy = journal_db.copy()
            journal_db_copy["Timestamp"] = pd.to_datetime(journal_db_copy["Timestamp"], errors="coerce")
            journal_db_copy = journal_db_copy.dropna(subset=["Timestamp"])

            if not journal_db_copy.empty:
                # Extract date (keep only date part, ignore time)
                journal_db_copy["Date"] = journal_db_copy["Timestamp"].dt.date
                # Get all dates with records (deduplicate)
                unique_dates = set(journal_db_copy["Date"].tolist())

                # Count from today backwards, calculate consecutive days
                today = datetime.date.today()
                streak = 0
                current_date = today

                # Check if today has a record
                while current_date in unique_dates:
                    streak += 1
                    current_date = current_date - datetime.timedelta(days=1)

                # If today has no record, start counting from yesterday
                if streak == 0:
                    current_date = today - datetime.timedelta(days=1)
                    while current_date in unique_dates:
                        streak += 1
                        current_date = current_date - datetime.timedelta(days=1)

                current_streak = streak
            else:
                current_streak = 0
        except Exception as e:
            print(f"⚠️ Error calculating streak: {e}")
            current_streak = 0
    else:
        current_streak = 0

    return {
        "total_actions": total_actions,
        "avg_energy": avg_energy,
        "current_streak": current_streak
    }


load_dotenv()
st.set_page_config(page_title="Momentum", page_icon="🧠", layout="wide")

# Initialize journal database (Session State) - Load from CSV or create new
if "journal_db" not in st.session_state:
    df_loaded = load_mind_flow_db()
    # All data in mind_flow_db.csv is journal logs, no need to filter by type
    if not df_loaded.empty:
        st.session_state.journal_db = df_loaded[["Timestamp", "Mood", "Energy", "Note"]].copy()
    else:
        st.session_state.journal_db = pd.DataFrame(columns=["Timestamp", "Mood", "Energy", "Note"])

# CSS Optimization (cleaner interface + message color blocks)
st.markdown("""
<style>
    .stChatMessage { font-family: 'Helvetica', sans-serif; }
    .stButton button { border-radius: 20px; }

    /* Shared message card styles */
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
        background-color: #e3f2fd;  /* Light blue */
        color: #0d47a1;
    }
    .mf-agent {
        background-color: #e8f5e9;  /* Light green */
        color: #1b5e20;
    }

    /* Supervisor Chain-of-Thought card (gray) */
    .mf-cot {
        background-color: #f5f5f5;  /* Light gray */
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

# --- 2. Sidebar: Settings and Data Dashboard ---
with st.sidebar:
    # === Quantified Self Dashboard (Top Metrics) ===
    st.header("📊 Quantified Self")
    metrics = calculate_dashboard_metrics()

    # Use columns to display three key metrics
    # Order: Streak (left), Avg Energy (middle), Actions (right)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Streak", metrics['current_streak'])
    with col2:
        st.metric("Avg Energy", f"{metrics['avg_energy']:.1f}")
    with col3:
        st.metric("Actions", metrics["total_actions"])

    st.divider()

    # API Key Management (Priority: Environment Variable > Secrets > Manual Input)
    # 1. First try to read from environment variable (loaded via load_dotenv() from .env file)
    api_key = os.getenv("GOOGLE_API_KEY")

    # 2. If environment variable not found, try to read from Streamlit Secrets
    if not api_key:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
        except StreamlitSecretNotFoundError:
            pass  # secrets.toml doesn't exist, continue to next step

    # 3. If neither exists, use manual input
    if not api_key:
        api_key = st.text_input("Google API Key", type="password", help="Please enter Gemini API Key")

    # === Navigation System ===
    st.markdown("### 🧭 Navigation System")

    # Load user profile from JSON file
    user_profile = load_user_profile()

    if user_profile.get("vision"):
        with st.container(border=True):
            st.caption("🔭 12-Week Vision")
            st.markdown(f"**{user_profile['vision']}**")

            st.divider()

            st.caption("⚙️ Daily System")
            st.markdown(f"**{user_profile['system']}**")
    else:
        with st.container(border=True):
            st.warning("System not yet established. Please interact with Strategist to set your 12-week vision!")

    st.divider()

    # === Debug Options ===
    # Debug: Display user_profile status
    if st.checkbox("🔍 Show Debug Info", False):
        user_profile = load_user_profile()
        st.write("**User Profile Status:**")
        st.json(user_profile)
        st.write("**Journal Data Status:**")
        st.write(f"- Session State Record Count: {len(st.session_state.journal_db)}")
        df_csv = load_mind_flow_db()
        st.write(f"- CSV File Record Count: {len(df_csv)}")
        st.write(f"- CSV File Path: {MIND_FLOW_DB_PATH}")
        if st.button("🗑️ Clear Conversation History (for testing)"):
            if "messages" in st.session_state:
                del st.session_state.messages
            st.rerun()

    st.divider()

    # === Safety Notice ===
    st.info(
        "**Safety Notice:** This is an AI coach, not a professional therapist. "
        "If you are in a crisis situation, please seek professional help immediately.",
        icon="⚠️"
    )

if not api_key:
    st.warning("Please enter API Key first to start Momentum.")
    st.stop()

# --- 3. Initialize Brain ---
# Create callback function to update journal


def update_journal(mood: str, energy: int, note: str):
    """Callback function to update journal database (updates both session_state and CSV)"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    new_entry = {
        "Timestamp": timestamp,
        "Mood": mood,
        "Energy": energy,
        "Note": note
    }
    # Update session_state
    st.session_state.journal_db = pd.concat(
        [st.session_state.journal_db, pd.DataFrame([new_entry])],
        ignore_index=True
    )
    # Also save to CSV file (ensure persistence)
    try:
        result = save_to_mind_flow_db(timestamp, mood, energy, note)
        if result is None:
            # Save failed, but session_state has been updated, so at least visible in this session
            print("⚠️ Warning: Journal entry updated to session_state, but failed to save to CSV")
    except Exception as e:
        print(f"⚠️ Error occurred while saving journal: {e}")
        # Even if save fails, continue execution, at least data is in session_state


# Use session_state to cache brain instance, avoid recreating each time
if "mind_flow_app" not in st.session_state:
    st.session_state.mind_flow_app = create_mind_flow_brain(
        api_key=api_key,
        model="gemini-2.0-flash",
        update_callback=update_journal
    )

# --- 4. User Interface (UX) ---

st.title("Momentum")
st.caption("Turn 12-Week Goals into Daily Systems | Your AI Companion for Executive Function.")

# Create main tabs: Chat / Dashboard
tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard"])

with tab_chat:
    # --- Quick Suggestion Buttons (placed at top of Chat tab) ---
    suggestions = ["🎯 Set Goal", "😫 I'm Anxious", "🐢 Low Motivation", "✅ Log Completion"]
    cols = st.columns(4)
    selected_prompt = None

    if cols[0].button("🎯 Set Goal", use_container_width=True):
        selected_prompt = suggestions[0]
    if cols[1].button("😫 I'm Anxious", use_container_width=True):
        selected_prompt = suggestions[1]
    if cols[2].button("🐢 Low Motivation", use_container_width=True):
        selected_prompt = suggestions[2]
    if cols[3].button("✅ Log Completion", use_container_width=True):
        selected_prompt = suggestions[3]

    # Create a container to hold history messages, ensuring it always displays above input box
    history_container = st.container()

    # Initialize conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

        # Decide which Agent to use based on user_profile status
        from brain import get_strategist_greeting, get_returning_user_greeting
        # Load user profile from JSON file
        user_profile = load_user_profile()

        # Check if onboarding is complete (system is set)
        if user_profile.get("system"):
            # Returning user: Use Starter (action) or Healer (care) directly
            # Default to Starter (action mode), can change to "healer" if Healer is needed
            with st.spinner("🚀 Starter is preparing greeting (returning user mode)..."):
                greeting_response = get_returning_user_greeting(
                    api_key=api_key,
                    model="gemini-2.0-flash",
                    plan_state=user_profile,
                    agent_type="starter"  # or "healer" for care mode
                )
        else:
            # New user or onboarding incomplete: Use Strategist
            with st.spinner("🧠 Strategist is preparing greeting..."):
                greeting_response = get_strategist_greeting(
                    api_key=api_key,
                    model="gemini-2.0-flash",
                    plan_state=user_profile
                )

        st.session_state.messages.append(greeting_response)

    # --- Input Area (bottom of Chat tab) ---

    # Get user input first
    user_input = st.chat_input("Tell me how you're feeling right now...")

    # Determine the actual text to send to Agent this round: prioritize chat_input, then quick buttons above
    prompt = user_input or selected_prompt

    # Input processing: only update state (messages, sidebar, etc.)
    if prompt:
        # 0.5 Input Guardrail: Validate user input
        is_valid, error_msg = input_guard(prompt)
        if not is_valid:
            st.warning(f"⚠️ {error_msg}")
            # Don't process invalid input
            prompt = None

        if prompt:
            # 1. Add User Message
            user_msg = HumanMessage(content=prompt)
            st.session_state.messages.append(user_msg)

            # 1.5 Safety check: self-harm/life-threatening keywords (hard guardrail)
            lowered = prompt.lower()
            if any(keyword in lowered for keyword in SAFETY_KEYWORDS):
                # Reply directly with fixed template, don't enter brain/don't call any tools
                safety_ai_message = AIMessage(content=SAFETY_MESSAGE)
                st.session_state.messages.append(safety_ai_message)
                st.warning("⚠️ Safety guardrail mechanism triggered, this conversation round will not enter Momentum brain.")
            else:
                # 2. Execute Agent (use lightweight prompt, not full-page blurry spinner)
                status = st.empty()
                status.markdown("⏳ Momentum team is collaborating...")
                result = st.session_state.mind_flow_app.invoke({"messages": st.session_state.messages})
                response = result["messages"][-1]
                status.empty()

                # 2.5 Output Guardrail: Clean AI response before storing/displaying
                if hasattr(response, 'content') and response.content:
                    cleaned_content = output_guard(response.content)
                    # Update response content with cleaned version
                    response.content = cleaned_content

                # 3. Add AI Response
                st.session_state.messages.append(response)

                # 3.5 Record this round's Supervisor reasoning result, for rendering to correspond to this response
                if "cot_history" not in st.session_state:
                    st.session_state.cot_history = []
                # The index of this AI response is the last one
                ai_index = len(st.session_state.messages) - 1
                st.session_state.cot_history.append({"idx": ai_index, "result": result})

                # 4. If there's a Tool Call, show success notification
                has_set_full_plan = False
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Check which tool was called
                    for tool_call in response.tool_calls:
                        tool_name = getattr(tool_call, 'name', None) or (tool_call.get('name') if isinstance(tool_call, dict) else None)
                        if tool_name == "save_journal_entry":
                            st.toast("✨ Journal entry written to database! Check sidebar data.", icon="✅")
                        elif tool_name == "set_full_plan":
                            has_set_full_plan = True
                            st.toast("✨ Plan created! Check sidebar navigation system.", icon="🎯")
                # 5. If any tool called set_full_plan this round (whether demo or regular conversation), immediately rerun to update sidebar
                if has_set_full_plan:
                    st.rerun()

    # Render history messages and RLHF feedback in history_container, ensuring they're always above input box
    with history_container:
        # Display history messages (including user/agent added this round), and record the last User / Agent pair
        last_user_msg = None
        last_agent_msg = None
        for idx, msg in enumerate(st.session_state.messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg
                render_message(msg)
            elif isinstance(msg, AIMessage):
                last_agent_msg = msg
                # First display Supervisor reasoning result corresponding to this idx (gray block above response)
                if "cot_history" in st.session_state:
                    for entry in st.session_state.cot_history:
                        if entry.get("idx") == idx:
                            render_supervisor_cot(entry.get("result"))
                            break
                # Then display Agent response itself
                render_message(msg)
            else:
                # Other message types (safety measure)
                render_message(msg)

        # RLHF feedback buttons (only shown for last Agent response, placed at bottom right of Agent block)
        if last_user_msg is not None and last_agent_msg is not None:
            # Maintain corresponding feedback status based on current last Agent message index, avoid cross-round residue
            if "feedback_status" not in st.session_state:
                st.session_state.feedback_status = {}
            if "last_agent_index" not in st.session_state:
                st.session_state.last_agent_index = None

            # If this round's last Agent index is different from previous round, reset this round's status
            current_agent_index = len(st.session_state.messages) - 1
            if st.session_state.last_agent_index != current_agent_index:
                st.session_state.last_agent_index = current_agent_index
                st.session_state.feedback_status[current_agent_index] = None

            current_status = st.session_state.feedback_status.get(current_agent_index)

            # Layout: three columns, first two blank, last two are adjacent thumbs up / thumbs down buttons (closer together)
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

            # Small hint text right below buttons, only shown for this round's Agent
            if current_status == "up":
                st.caption("🙏 Recorded this response as 'helpful'")
            elif current_status == "down":
                st.caption("📥 Recorded this response as 'not helpful'")

with tab_dashboard:
    st.subheader("📊 Flow Journal")
    if not st.session_state.journal_db.empty:
        st.write("Last 7 journal entries:")
        st.dataframe(st.session_state.journal_db.tail(7), hide_index=True)

        st.write("Energy Index Trend (Last 7 Days):")
        # Prepare chart data: filter last 7 days of data and group by date to calculate average
        try:
            journal_db_copy = st.session_state.journal_db.copy()
            journal_db_copy["Timestamp"] = pd.to_datetime(journal_db_copy["Timestamp"], errors="coerce")
            journal_db_copy["Energy"] = pd.to_numeric(journal_db_copy["Energy"], errors="coerce")

            # Filter last 7 days of data
            seven_days_ago = datetime.datetime.now() - datetime.timedelta(days=7)
            recent_data = journal_db_copy[journal_db_copy["Timestamp"] >= seven_days_ago].copy()
            recent_data = recent_data.dropna(subset=["Timestamp", "Energy"])

            if not recent_data.empty:
                # Convert timestamp to date (keep only date part)
                recent_data["Date"] = recent_data["Timestamp"].dt.date

                # Group by date, calculate average energy value per day
                daily_avg = recent_data.groupby("Date", as_index=False).agg({
                    "Energy": "mean"
                })
                daily_avg["Energy"] = daily_avg["Energy"].round(1)  # Keep one decimal place

                # Convert date back to datetime type for Altair use
                daily_avg["Date"] = pd.to_datetime(daily_avg["Date"])
                daily_avg = daily_avg.sort_values("Date")

                # Use Altair to create chart, set y-axis max to 10
                chart = alt.Chart(daily_avg).mark_line(point=True).encode(
                    x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%Y-%m-%d', labelAngle=-45)),
                    y=alt.Y('Energy:Q', title='Energy Index', scale=alt.Scale(domain=[0, 10]))
                ).properties(
                    width='container',
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No energy data in the last 7 days.")
        except Exception as e:
            st.warning(f"Unable to draw trend chart: {str(e)}")
    else:
        st.info("No journal data yet. It will be automatically recorded after completing an action.")
