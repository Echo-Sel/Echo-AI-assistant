import json
import os
from datetime import datetime
from langchain.tools import tool

TODO_FILE = os.getenv("TODO_FILE", "todo_list.json")

def load_todo_list():
    if not os.path.exists(TODO_FILE):
        return {"tasks": []}
    try:
        with open(TODO_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"tasks": []}

def save_todo_list(data):
    with open(TODO_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_next_id(tasks):
    if not tasks:
        return 1
    return max(task["id"] for task in tasks) + 1

@tool
def add_task(task: str, priority: str = "medium", due_date: str = None):
    """Adds a new task to the to-do list.
    Args:
        task: The description of the task (e.g., "Buy groceries").
        priority: The priority level ("high", "medium", "low"). Defaults to "medium".
        due_date: Optional due date string (e.g., "2025-11-30" or "tomorrow").
    """
    data = load_todo_list()
    new_task = {
        "id": get_next_id(data["tasks"]),
        "task": task,
        "status": "incomplete",
        "priority": priority.lower(),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "due_date": due_date
    }
    data["tasks"].append(new_task)
    save_todo_list(data)
    return f"I've added '{task}' to your to-do list with {priority} priority."

@tool
def list_tasks(status: str = "incomplete", priority: str = None):
    """Lists tasks from the to-do list.
    Args:
        status: Filter by status ("incomplete", "complete", or "all"). Defaults to "incomplete".
        priority: Optional. Filter by priority ("high", "medium", "low").
    """
    data = load_todo_list()
    tasks = data["tasks"]
    
    if status != "all":
        tasks = [t for t in tasks if t["status"] == status]
    
    if priority:
        tasks = [t for t in tasks if t["priority"] == priority.lower()]
        
    if not tasks:
        return f"You have no {status} tasks{' with ' + priority + ' priority' if priority else ''}."
        
    result = f"Here are your {status} tasks:\n"
    for t in tasks:
        due_str = f" (Due: {t['due_date']})" if t.get("due_date") else ""
        prio_str = f" [{t['priority'].upper()}]"
        result += f"{t['id']}. {t['task']}{prio_str}{due_str}\n"
    return result

@tool
def complete_task(task_name: str):
    """Marks a task as complete.
    Args:
        task_name: The name (or part of the name) of the task to complete.
    """
    data = load_todo_list()
    # Simple fuzzy match: check if task_name is in the task description
    matches = [t for t in data["tasks"] if task_name.lower() in t["task"].lower() and t["status"] == "incomplete"]
    
    if not matches:
        return f"I couldn't find any incomplete task matching '{task_name}'."
    
    if len(matches) > 1:
        return f"I found multiple tasks matching '{task_name}': {', '.join([t['task'] for t in matches])}. Please be more specific."
        
    target_task = matches[0]
    target_task["status"] = "complete"
    target_task["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_todo_list(data)
    return f"I've marked '{target_task['task']}' as complete."

@tool
def delete_task(task_name: str):
    """Permanently deletes a task from the list.
    Args:
        task_name: The name (or part of the name) of the task to delete.
    """
    data = load_todo_list()
    matches = [t for t in data["tasks"] if task_name.lower() in t["task"].lower()]
    
    if not matches:
        return f"I couldn't find any task matching '{task_name}'."
        
    if len(matches) > 1:
        return f"I found multiple tasks matching '{task_name}': {', '.join([t['task'] for t in matches])}. Please be more specific."
        
    target_task = matches[0]
    data["tasks"].remove(target_task)
    save_todo_list(data)
    return f"I've removed '{target_task['task']}' from your to-do list."

@tool
def clear_completed_tasks():
    """Removes all completed tasks from the list."""
    data = load_todo_list()
    original_count = len(data["tasks"])
    data["tasks"] = [t for t in data["tasks"] if t["status"] == "incomplete"]
    removed_count = original_count - len(data["tasks"])
    save_todo_list(data)
    return f"I've cleared {removed_count} completed tasks from your list."
