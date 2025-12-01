"""
Momentum Test - 終端機測試腳本
用於快速測試大腦邏輯，無需啟動 Streamlit 介面
"""
import os
import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from brain import create_mind_flow_brain, get_strategist_greeting, get_returning_user_greeting, load_user_profile

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
    "⚠️ 我注意到你提到可能與自我傷害或生命安全有關的內容。\n"
    "我是一個 AI，沒有醫療或心理專業資格，也無法在緊急狀況中提供即時協助。\n\n"
    "👉 如果你有**立即的危險**，請立刻聯絡你所在地的緊急電話（例如 911），\n"
    "或撥打當地的自殺防治／心理諮詢專線，並尋求家人、朋友或信任的人陪伴你。\n\n"
    "你值得被好好對待，也值得被真正看見和幫助。"
)


class ConversationLogger:
    """對話記錄器，將對話內容保存到文件"""
    
    def __init__(self):
        # 創建 logs 目錄（如果不存在）
        self.logs_dir = "logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # 創建日誌文件，文件名包含時間戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f"conversation_{timestamp}.txt")
        self.file = open(self.log_file, "w", encoding="utf-8")
        
        # 寫入開始標記
        self.write_separator()
        self.write(f"🧠 Momentum 對話記錄")
        self.write(f"開始時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_separator()
        self.file.flush()
    
    def write(self, text):
        """寫入文本到文件和終端"""
        print(text)
        self.file.write(text + "\n")
        self.file.flush()
    
    def write_separator(self):
        """寫入分隔線"""
        separator = "=" * 50
        print(separator)
        self.file.write(separator + "\n")
        self.file.flush()
    
    def close(self):
        """關閉文件"""
        self.write_separator()
        self.write(f"結束時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_separator()
        self.file.close()
        print(f"\n💾 對話記錄已保存到: {self.log_file}")


def main():
    """主測試循環"""
    # 初始化對話記錄器
    logger = ConversationLogger()
    
    try:
        # 載入環境變數
        load_dotenv()
        
        # 獲取 API Key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("請輸入 Google API Key: ").strip()
            if not api_key:
                logger.write("❌ 需要 API Key 才能運行")
                return
        
        logger.write("🧠 Momentum - 終端測試模式")
        logger.write_separator()
        logger.write("輸入 'quit' 或 'exit' 退出\n")
        
        # 創建大腦（不使用 journal_db，因為終端測試不需要持久化）
        app = create_mind_flow_brain(api_key=api_key, model="gemini-2.0-flash")
        
        # 初始化對話 - 根據 user_profile 狀態決定使用哪個 Agent
        messages = []
        user_profile = load_user_profile()
        
        # 檢查是否已完成 onboarding（vision 和 system 都已設置）
        if user_profile.get("vision") and user_profile.get("system"):
            # 老用戶：直接使用 Starter（啟動）或 Healer（關心）
            # 預設使用 Starter（啟動模式），如果需要 Healer 可以改為 "healer"
            logger.write("🚀 Starter 正在準備問候（老用戶模式）...\n")
            greeting_response = get_returning_user_greeting(
                api_key=api_key,
                model="gemini-2.0-flash",
                plan_state=user_profile,
                agent_type="starter"  # 或 "healer" 用於關心模式
            )
        else:
            # 新用戶或未完成 onboarding：使用 Strategist
            logger.write("🧠 Strategist 正在準備問候...\n")
            greeting_response = get_strategist_greeting(
                api_key=api_key,
                model="gemini-2.0-flash",
                plan_state=user_profile
            )
        
        logger.write(f"🤖 {greeting_response.content}\n")
        messages.append(greeting_response)
        
        # 對話循環
        while True:
            # 獲取用戶輸入（input 會自動顯示提示符，不需要重複打印）
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                logger.write("\n👋 再見！")
                break
            
            # 記錄用戶輸入到日誌文件（不重複打印到終端，因為 input 已經顯示了）
            logger.file.write(f"👤 你: {user_input}\n")
            logger.file.flush()

            # --- 安全檢查：自我傷害／生命危險關鍵字 ---
            lowered = user_input.lower()
            if any(keyword in lowered for keyword in SAFETY_KEYWORDS):
                # 直接回覆固定的安全訊息，不進入大腦／不執行任何工具
                logger.write("\n⚠️ [安全守門機制觸發 - 跳過大腦路由與工具調用]\n")
                logger.write(f"🤖 {SAFETY_MESSAGE}\n")
                # 不將這輪輸入送入 LangGraph，以避免被當作一般對話處理
                continue

            # 添加用戶訊息
            messages.append(HumanMessage(content=user_input))
            
            # 執行大腦
            logger.write("\n🤔 Momentum 團隊正在協作中...\n")
            try:
                result = app.invoke({"messages": messages})
                
                # 調試：顯示 supervisor 推理過程和路由信息
                if result.get("reasoning"):
                    logger.write("\n💭 [Supervisor 推理過程 (Chain-of-Thought)]")
                    logger.write("-" * 50)
                    # 按行打印推理過程，保持格式
                    for line in result['reasoning'].split('\n'):
                        if line.strip():  # 只打印非空行
                            logger.write(f"  {line.strip()}")
                    logger.write("-" * 50 + "\n")
                
                if result.get("debug_info"):
                    logger.write(f"{result['debug_info']}\n")
                
                # 檢查所有消息，找到最後的 AIMessage（可能包含工具調用或文本回應）
                # 因為最後一個可能是 ToolMessage，需要找到最後的 AIMessage
                response = None
                tool_call_message = None
                
                # 從後往前找最後的 AIMessage
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        if response is None:
                            response = msg  # 最後的 AIMessage
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            tool_call_message = msg
                
                # 如果找不到 AIMessage，使用最後一個消息
                if response is None:
                    response = result["messages"][-1]
                
                # 檢查 response.content 是否包含 tool_code 標記（這是錯誤的格式）
                # 如果包含，說明 LLM 沒有正確使用工具調用，而是用文本格式描述了工具
                content_to_display = None
                if hasattr(response, 'content') and response.content:
                    # 過濾掉 tool_code 標記的內容（這是錯誤的格式）
                    if '```tool_code' in response.content or 'tool_code' in response.content.lower():
                        # 這是錯誤的格式，不應該顯示，應該顯示工具調用的信息
                        # 但如果有真正的工具調用，會在下麵顯示
                        content_to_display = None
                    else:
                        content_to_display = response.content
                
                # 顯示回應（如果有有效的文本內容，且不是工具調用的錯誤格式）
                if content_to_display:
                    logger.write(f"🤖 {content_to_display}\n")
                
                # 如果有工具調用，顯示詳細信息
                if tool_call_message and tool_call_message.tool_calls:
                    for tool_call in tool_call_message.tool_calls:
                        # 獲取工具名稱
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get('name', '')
                            args = tool_call.get('args', {})
                        else:
                            tool_name = getattr(tool_call, 'name', '')
                            args = getattr(tool_call, 'args', {})
                        
                        if tool_name == "set_full_plan":
                            # 如果是 set_full_plan，打印詳細內容供確認
                            logger.write("\n📋 [Strategist 工具調用 - 計劃內容確認]\n")
                            logger.write(f"🔭 Vision (12週目標): {args.get('vision', 'N/A') if isinstance(args, dict) else 'N/A'}\n")
                            logger.write(f"⚙️  System (每日習慣): {args.get('system', 'N/A') if isinstance(args, dict) else 'N/A'}\n")
                            logger.write("💡 注意：Starter 會根據當前狀態動態生成微行動建議\n")
                            logger.write("=" * 50 + "\n")
                        else:
                            logger.write(f"✨ [工具已執行: {tool_name}]\n")
                
                # 更新訊息歷史
                messages.append(response)
                
            except Exception as e:
                logger.write(f"❌ 錯誤: {e}\n")
                # 移除最後的用戶訊息，以便重試
                messages.pop()
    
    finally:
        # 確保關閉日誌文件
        logger.close()


if __name__ == "__main__":
    main()

