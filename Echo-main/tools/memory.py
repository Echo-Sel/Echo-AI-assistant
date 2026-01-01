import json
import os
from datetime import datetime
from langchain.tools import tool

MEMORY_FILE = os.getenv("MEMORY_FILE", "echo_memory.json")

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"user": {}}
    try:
        with open(MEMORY_FILE, 'r') as f:
            data = json.load(f)
        
        # Migration logic: Check if it's the old flat format
        # Old format: keys map directly to memory objects with "value" and "timestamp"
        # New format: keys map to profile dicts, which then map to memory objects
        
        if not data:
            return {"user": {}}
            
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict) and "value" in first_val and "timestamp" in first_val:
            # This is the old format, migrate it to "user" profile
            return {"user": data}
            
        return data
    except json.JSONDecodeError:
        return {"user": {}}

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=4)

@tool
def remember_fact(key: str, value: str, profile: str = "user"):
    """Saves a new piece of information or fact about a specific person or context.
    Use this when the user tells you something personal, a preference, or asks you to remember something.
    
    Args:
        key: A short, descriptive key for the memory (e.g., 'name', 'favorite_color', 'birthday').
        value: The information to remember.
        profile: The profile to associate this memory with (e.g., 'user', 'mom', 'dad', 'brother', 'dog'). Defaults to 'user'.
    """
    memory = load_memory()
    
    if profile not in memory:
        memory[profile] = {}
        
    memory[profile][key] = {
        "value": value,
        "timestamp": datetime.now().isoformat()
    }
    save_memory(memory)
    return f"I'll remember that {profile}'s {key.replace('_', ' ')} is {value}."

@tool
def recall_fact(key: str, profile: str = "user"):
    """Retrieves a previously saved fact or piece of information about a specific person.
    Use this when the user asks about something they told you before.
    
    Args:
        key: The key of the memory to retrieve (e.g., 'name').
        profile: The profile to retrieve the memory from (e.g., 'user', 'mom', 'dad'). Defaults to 'user'.
    """
    memory = load_memory()
    
    if profile in memory and key in memory[profile]:
        return f"According to my memory, {profile}'s {key.replace('_', ' ')} is {memory[profile][key]['value']}."
    else:
        return f"I don't have any memory stored for {profile}'s '{key}'."

@tool
def list_all_memories(profile: str = None):
    """Lists all the facts and information currently stored in memory.
    Use this when the user asks 'what do you remember?' or 'what do you know about me?'.
    
    Args:
        profile: Optional. If provided, lists memories only for that profile. If not provided, lists all memories for all profiles.
    """
    memory = load_memory()
    if not memory:
        return "I don't have any memories stored yet."
    
    result = ""
    
    if profile:
        if profile in memory and memory[profile]:
            result += f"Memories for {profile}:\n"
            for key, data in memory[profile].items():
                result += f"- {key.replace('_', ' ')}: {data['value']}\n"
        else:
            return f"I don't have any memories stored for {profile}."
    else:
        result = "Here is everything I remember:\n"
        for prof, items in memory.items():
            if items:
                result += f"\n[{prof.upper()}]\n"
                for key, data in items.items():
                    result += f"- {key.replace('_', ' ')}: {data['value']}\n"
    
    return result

@tool
def forget_fact(key: str, profile: str = "user"):
    """Deletes a specific fact or piece of information from memory.
    Use this when the user asks you to forget something.
    
    Args:
        key: The key of the memory to delete.
        profile: The profile to delete the memory from. Defaults to 'user'.
    """
    memory = load_memory()
    
    if profile in memory and key in memory[profile]:
        del memory[profile][key]
        # Clean up empty profile if needed, but keeping it is fine too.
        save_memory(memory)
        return f"I have forgotten {profile}'s {key.replace('_', ' ')}."
    else:
        return f"I couldn't find any memory for {profile}'s '{key}' to forget."
