import os
import re
import logging
import time
import subprocess
from dotenv import load_dotenv
import speech_recognition as sr
from langchain_ollama import ChatOllama, OllamaLLM

# from langchain_openai import ChatOpenAI # if you want to use openai
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# importing tools
from tools.time import get_time, get_current_date # added "get_current_date from time.py 17/11/25"
from tools.OCR import read_text_from_latest_image
from tools.arp_scan import arp_scan_terminal
from tools.duckduckgo import duckduckgo_search_tool
from tools.matrix import matrix_mode
from tools.screenshot import take_screenshot
from tools.memory import remember_fact, recall_fact, list_all_memories, forget_fact, load_memory
from tools.todo import add_task, list_tasks, complete_task, delete_task, clear_completed_tasks
from tools.calculator import calculate

load_dotenv()

MIC_INDEX = None
TRIGGER_WORD = "echo"
CONVERSATION_TIMEOUT = 30  # seconds of inactivity before exiting conversation mode

logging.basicConfig(level=logging.INFO)  # logging


recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=MIC_INDEX)

# Initialize LLM (changed model from qwen3:1.7b to llama3.2b for more power and better tool calling "17/11/25")
llm = ChatOllama(model="llama3.2:3b", keep_alive="24h", reasoning=False)

# llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, organization=org_id) for openai

# Tool list (updated to account for the change in time.py 17/11/25)
tools = [get_time, get_current_date, arp_scan_terminal, read_text_from_latest_image, duckduckgo_search_tool, matrix_mode, take_screenshot, remember_fact, recall_fact, list_all_memories, forget_fact, calculate, add_task, list_tasks, complete_task, delete_task, clear_completed_tasks]
# Tool-calling prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Echo, an intelligent AI assistant with access to tools. "
            "CRITICAL RULES: "
            "- When users ask 'what day is it', 'what's the date', or 'what time is it' (without specifying a city), you MUST call the get_current_date tool. "
            "- When users ask for time in a specific city like 'what time is it in London', call the get_time tool with that city name. "
            "- NEVER respond about time or date without calling one of these tools first. "
            "- When users ask for a calculation, use the 'calculate' tool. "
            "- MEMORY RULES: "
            "- Use 'remember_fact' to save info. Identify the subject (User, Mom, Dad, Brother, Dog, Best Friend, etc.) and pass it as the 'profile' argument (lowercase). Default to 'user' if about the speaker. "
            "- Use 'recall_fact' to retrieve info. Specify the 'profile' if asking about someone else (e.g., 'what is my mom's name?' -> profile='mom', 'what is my best friend's name?' -> profile='best friend'). DO NOT PASS A 'VALUE' ARGUMENT TO RECALL_FACT. "
            "- Use 'list_all_memories' to show what you know. You can specify a profile to filter. "
            "- Use 'forget_fact' to delete specific memories for a profile. "
            "- TO-DO LIST RULES: "
            "- Use 'add_task' to add new tasks. You can specify priority (high/medium/low) and due date. "
            "- Use 'list_tasks' to see what needs to be done. You can filter by status='incomplete' (default) or 'all', and by priority. "
            "- Use 'complete_task' to mark a task as done. "
            "- Use 'delete_task' to remove a task entirely. "
            "- Use 'clear_completed_tasks' to remove all finished tasks. "
            "- After calling a tool, present the information in a natural, conversational way. "
            "Always use available tools to provide accurate information.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# Agent + executor
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# TTS setup
def speak_text(text: str):
    try:
        subprocess.run(["say", "-v", "Ava", text])
    except Exception as e:
        logging.error(f"❌ TTS failed: {e}")


def solve_simple_math(text: str):
    """
    Attempts to solve simple math problems locally using regex and eval.
    Returns the result string if successful, or None if it's not a simple math problem.
    """
    # Normalize text: replace "x" or "times" with "*", "plus" with "+", "minus" with "-", "divided by" with "/"
    text = text.lower()
    text = text.replace("times", "*").replace(" x ", " * ").replace("plus", "+").replace("minus", "-").replace("divided by", "/")
    
    # Regex to find simple math expressions like "5 + 5", "10 * 20", "100 / 5"
    # Looks for: number, operator, number
    match = re.search(r'(\d+(\.\d+)?)\s*([\+\-\*\/])\s*(\d+(\.\d+)?)', text)
    
    if match:
        try:
            expression = match.group(0)
            result = eval(expression)
            # Check if result is an integer (e.g. 5.0 -> 5)
            if result == int(result):
                result = int(result)
            return f"The result is {result}"
        except:
            return None
    return None

def check_local_intent(text: str):
    """
    Checks for common memory queries and handles them locally to bypass the LLM.
    """
    text = text.lower().strip(".,!?")
    
    # Pattern 1: "what is my [key]" -> profile="user" OR profile=[key]
    # e.g. "what is my name" -> key="name", profile="user"
    # e.g. "what is my mom" -> key="mom" -> profile="mom"
    # e.g. "what is my best friend" -> key="best friend" -> profile="best friend"
    match = re.search(r"what(?:'s| is) my ([\w\s]+)$", text)
    if match:
        key = match.group(1).strip()
        
        # Check if the key is actually a profile (e.g., "mom", "dad", "best friend")
        memory = load_memory()
        if key in memory:
            logging.info(f"🧠 Local memory check: detected profile='{key}'")
            return list_all_memories.invoke({"profile": key})
            
        logging.info(f"🧠 Local memory check: key='{key}', profile='user'")
        return recall_fact.invoke({"key": key, "profile": "user"})
        
    # Pattern 2: "what is my [profile]'s [key]"
    # e.g. "what is my mom's name", "what's my dog's breed"
    # e.g. "what is my best friend's name"
    match = re.search(r"what(?:'s| is) my ([\w\s]+)'s (\w+)$", text)
    if match:
        profile = match.group(1).strip()
        key = match.group(2).strip()
        logging.info(f"🧠 Local memory check: key='{key}', profile='{profile}'")
        return recall_fact.invoke({"key": key, "profile": profile})
        
    return None

# Main interaction loop
def write():
    conversation_mode = False
    last_interaction_time = None

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            while True:
                try:
                    if not conversation_mode:
                        logging.info("🎤 Listening for wake word...")
                        audio = recognizer.listen(source, timeout=10)
                        transcript = recognizer.recognize_google(audio)
                        logging.info(f"🗣 Heard: {transcript}")

                        if TRIGGER_WORD.lower() in transcript.lower():
                            logging.info(f"🗣 Triggered by: {transcript}")
                            speak_text("Yes sir?")
                            conversation_mode = True
                            last_interaction_time = time.time()
                        else:
                            logging.debug("Wake word not detected, continuing...")
                    else:
                        logging.info("🎤 Listening for next command...")
                        audio = recognizer.listen(source, timeout=10)
                        command = recognizer.recognize_google(audio)
                        logging.info(f"📥 Command: {command}")

                        # 1. Check for exit commands
                        exit_commands = ["no", "no thanks", "no thank you", "stop", "exit", "quit", "nothing", "that's all", "bye", "goodbye"]
                        # Check if command matches exactly or contains exit phrase at end
                        cmd_lower = command.lower().strip(".,!?")
                        if cmd_lower in exit_commands or any(cmd_lower.endswith(exit_cmd) for exit_cmd in exit_commands):
                            logging.info("🛑 Exit command received.")
                            speak_text("Alright, goodbye!")
                            conversation_mode = False
                            continue

                        # 2. Check for simple math (local processing)
                        math_result = solve_simple_math(command)
                        if math_result:
                            logging.info(f"🧮 Solved math locally: {math_result}")
                            print("Echo:", math_result)
                            speak_text(math_result)
                            last_interaction_time = time.time()
                            continue
                            
                        # 3. Check for local memory intent
                        local_memory_result = check_local_intent(command)
                        if local_memory_result:
                            logging.info(f"🧠 Local memory result: {local_memory_result}")
                            print("Echo:", local_memory_result)
                            speak_text(local_memory_result)
                            last_interaction_time = time.time()
                            continue

                        logging.info("🤖 Sending command to agent...")
                        response = executor.invoke({"input": command})
                        content = response["output"]
                        logging.info(f"✅ Agent responded: {content}")

                        print("Echo:", content)
                        speak_text(content)
                        last_interaction_time = time.time()

                        if time.time() - last_interaction_time > CONVERSATION_TIMEOUT:
                            logging.info("⌛ Timeout: Returning to wake word mode.")
                            conversation_mode = False

                except sr.WaitTimeoutError:
                    logging.warning("⚠️ Timeout waiting for audio.")
                    if (
                        conversation_mode
                        and time.time() - last_interaction_time > CONVERSATION_TIMEOUT
                    ):
                        logging.info(
                            "⌛ No input in conversation mode. Returning to wake word mode."
                        )
                        conversation_mode = False
                except sr.UnknownValueError:
                    logging.warning("⚠️ Could not understand audio.")
                except Exception as e:
                    logging.error(f"❌ Error during recognition or tool call: {e}")
                    time.sleep(1)

    except Exception as e:
        logging.critical(f"❌ Critical error in main loop: {e}")


if __name__ == "__main__":
    write()
