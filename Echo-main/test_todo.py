from tools.todo import add_task, list_tasks, complete_task, delete_task, clear_completed_tasks
import os
import json

# Use a temporary file for testing
os.environ["TODO_FILE"] = "test_todo_list.json"

def test_todo():
    print("--- Testing To-Do List Tools ---")
    
    # 1. Clear any existing test data
    if os.path.exists("test_todo_list.json"):
        os.remove("test_todo_list.json")
        
    # 2. Add Tasks
    print("\n--- Adding Tasks ---")
    print(add_task.invoke({"task": "Buy milk", "priority": "high"}))
    print(add_task.invoke({"task": "Walk the dog", "priority": "medium", "due_date": "today"}))
    print(add_task.invoke({"task": "Read a book", "priority": "low"}))
    
    # 3. List Tasks
    print("\n--- Listing Tasks (Incomplete) ---")
    print(list_tasks.invoke({"status": "incomplete"}))
    
    print("\n--- Listing Tasks (High Priority) ---")
    print(list_tasks.invoke({"status": "incomplete", "priority": "high"}))
    
    # 4. Complete Task
    print("\n--- Completing Task 'Buy milk' ---")
    print(complete_task.invoke({"task_name": "Buy milk"}))
    
    print("\n--- Listing Tasks (After Completion) ---")
    print(list_tasks.invoke({"status": "incomplete"}))
    
    # 5. Delete Task
    print("\n--- Deleting Task 'Read a book' ---")
    print(delete_task.invoke({"task_name": "Read a book"}))
    
    print("\n--- Final List ---")
    print(list_tasks.invoke({"status": "all"}))
    
    # 6. Clear Completed
    print("\n--- Clearing Completed Tasks ---")
    print(clear_completed_tasks.invoke({}))
    
    print("\n--- List After Clearing ---")
    print(list_tasks.invoke({"status": "all"}))
    
    # Cleanup
    if os.path.exists("test_todo_list.json"):
        os.remove("test_todo_list.json")

if __name__ == "__main__":
    test_todo()
