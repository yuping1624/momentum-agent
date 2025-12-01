"""
Momentum Test - Terminal Test Script
For quickly testing brain logic without starting Streamlit interface
"""
import os
import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from brain import create_mind_flow_brain, get_strategist_greeting, get_returning_user_greeting, load_user_profile

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
    "⚠️ I noticed you mentioned content that may be related to self-harm or life safety.\n"
    "I am an AI and do not have medical or psychological professional qualifications, "
    "and I cannot provide immediate assistance in emergency situations.\n\n"
    "👉 If you are in **immediate danger**, please contact your local emergency number (e.g., 911) immediately,\n"
    "or call your local suicide prevention/mental health hotline, and seek support from family, friends, or trusted people.\n\n"
    "You deserve to be treated well and to be truly seen and helped."
)


class ConversationLogger:
    """Conversation logger, saves conversation content to file"""
    
    def __init__(self):
        # Create logs directory (if it doesn't exist)
        self.logs_dir = "logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # Create log file, filename includes timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f"conversation_{timestamp}.txt")
        self.file = open(self.log_file, "w", encoding="utf-8")
        
        # Write start marker
        self.write_separator()
        self.write(f"🧠 Momentum Conversation Log")
        self.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_separator()
        self.file.flush()
    
    def write(self, text):
        """Write text to file and terminal"""
        print(text)
        self.file.write(text + "\n")
        self.file.flush()
    
    def write_separator(self):
        """Write separator line"""
        separator = "=" * 50
        print(separator)
        self.file.write(separator + "\n")
        self.file.flush()
    
    def close(self):
        """Close file"""
        self.write_separator()
        self.write(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_separator()
        self.file.close()
        print(f"\n💾 Conversation log saved to: {self.log_file}")


def main():
    """Main test loop"""
    # Initialize conversation logger
    logger = ConversationLogger()
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API Key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("Please enter Google API Key: ").strip()
            if not api_key:
                logger.write("❌ API Key required to run")
                return
        
        logger.write("🧠 Momentum - Terminal Test Mode")
        logger.write_separator()
        logger.write("Type 'quit' or 'exit' to exit\n")
        
        # Create brain (not using journal_db, as terminal test doesn't need persistence)
        app = create_mind_flow_brain(api_key=api_key, model="gemini-2.0-flash")
        
        # Initialize conversation - decide which Agent to use based on user_profile status
        messages = []
        user_profile = load_user_profile()
        
        # Check if onboarding is complete (vision and system are both set)
        if user_profile.get("vision") and user_profile.get("system"):
            # Returning user: Use Starter (action) or Healer (care) directly
            # Default to Starter (action mode), can change to "healer" if Healer is needed
            logger.write("🚀 Starter is preparing greeting (returning user mode)...\n")
            greeting_response = get_returning_user_greeting(
                api_key=api_key,
                model="gemini-2.0-flash",
                plan_state=user_profile,
                agent_type="starter"  # or "healer" for care mode
            )
        else:
            # New user or onboarding incomplete: Use Strategist
            logger.write("🧠 Strategist is preparing greeting...\n")
            greeting_response = get_strategist_greeting(
                api_key=api_key,
                model="gemini-2.0-flash",
                plan_state=user_profile
            )
        
        logger.write(f"🤖 {greeting_response.content}\n")
        messages.append(greeting_response)
        
        # Conversation loop
        while True:
            # Get user input (input will automatically display prompt, no need to print again)
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                logger.write("\n👋 Goodbye!")
                break
            
            # Log user input to log file (don't print to terminal again, as input already displayed it)
            logger.file.write(f"👤 You: {user_input}\n")
            logger.file.flush()

            # --- Safety check: self-harm/life-threatening keywords ---
            lowered = user_input.lower()
            if any(keyword in lowered for keyword in SAFETY_KEYWORDS):
                # Reply directly with fixed safety message, don't enter brain/don't execute any tools
                logger.write("\n⚠️ [Safety guardrail triggered - skipping brain routing and tool calls]\n")
                logger.write(f"🤖 {SAFETY_MESSAGE}\n")
                # Don't send this round's input to LangGraph, to avoid being treated as regular conversation
                continue

            # Add user message
            messages.append(HumanMessage(content=user_input))
            
            # Execute brain
            logger.write("\n🤔 Momentum team is collaborating...\n")
            try:
                result = app.invoke({"messages": messages})
                
                # Debug: display supervisor reasoning process and routing info
                if result.get("reasoning"):
                    logger.write("\n💭 [Supervisor Reasoning Process (Chain-of-Thought)]")
                    logger.write("-" * 50)
                    # Print reasoning process line by line, maintain format
                    for line in result['reasoning'].split('\n'):
                        if line.strip():  # Only print non-empty lines
                            logger.write(f"  {line.strip()}")
                    logger.write("-" * 50 + "\n")
                
                if result.get("debug_info"):
                    logger.write(f"{result['debug_info']}\n")
                
                # Check all messages, find the last AIMessage (may contain tool calls or text response)
                # Because the last one might be ToolMessage, need to find the last AIMessage
                response = None
                tool_call_message = None
                
                # Find last AIMessage from back to front
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        if response is None:
                            response = msg  # Last AIMessage
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            tool_call_message = msg
                
                # If AIMessage not found, use last message
                if response is None:
                    response = result["messages"][-1]
                
                # Check if response.content contains tool_code markers (this is wrong format)
                # If it contains, it means LLM didn't use tool calls correctly, but described tools in text format
                content_to_display = None
                if hasattr(response, 'content') and response.content:
                    # Filter out tool_code marked content (this is wrong format)
                    if '```tool_code' in response.content or 'tool_code' in response.content.lower():
                        # This is wrong format, shouldn't display, should display tool call info
                        # But if there are real tool calls, they will be displayed below
                        content_to_display = None
                    else:
                        content_to_display = response.content
                
                # Display response (if there's valid text content and not wrong format for tool calls)
                if content_to_display:
                    logger.write(f"🤖 {content_to_display}\n")
                
                # If there are tool calls, display detailed info
                if tool_call_message and tool_call_message.tool_calls:
                    for tool_call in tool_call_message.tool_calls:
                        # Get tool name
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get('name', '')
                            args = tool_call.get('args', {})
                        else:
                            tool_name = getattr(tool_call, 'name', '')
                            args = getattr(tool_call, 'args', {})
                        
                        if tool_name == "set_full_plan":
                            # If it's set_full_plan, print detailed content for confirmation
                            logger.write("\n📋 [Strategist Tool Call - Plan Content Confirmation]\n")
                            logger.write(f"🔭 Vision (12-week goal): {args.get('vision', 'N/A') if isinstance(args, dict) else 'N/A'}\n")
                            logger.write(f"⚙️  System (Daily habit): {args.get('system', 'N/A') if isinstance(args, dict) else 'N/A'}\n")
                            logger.write("💡 Note: Starter will dynamically generate micro-action suggestions based on current state\n")
                            logger.write("=" * 50 + "\n")
                        else:
                            logger.write(f"✨ [Tool executed: {tool_name}]\n")
                
                # Update message history
                messages.append(response)
                
            except Exception as e:
                logger.write(f"❌ Error: {e}\n")
                # Remove last user message to allow retry
                messages.pop()
    
    finally:
        # Ensure log file is closed
        logger.close()


if __name__ == "__main__":
    main()

