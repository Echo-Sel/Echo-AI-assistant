from tools.memory import remember_fact, recall_fact, list_all_memories, forget_fact
import json

def test_memory():
    print("--- Initial Memory (should trigger migration if old format) ---")
    print(list_all_memories.invoke({}))
    
    print("\n--- Adding Memories ---")
    print(remember_fact.invoke({"key": "name", "value": "Rania", "profile": "mom"}))
    print(remember_fact.invoke({"key": "hobby", "value": "Gardening", "profile": "mom"}))
    print(remember_fact.invoke({"key": "name", "value": "Leo", "profile": "dog"}))
    print(remember_fact.invoke({"key": "breed", "value": "Golden Retriever", "profile": "dog"}))
    
    print("\n--- Listing All Memories ---")
    print(list_all_memories.invoke({}))
    
    print("\n--- Recalling Specific Facts ---")
    print(recall_fact.invoke({"key": "name", "profile": "mom"}))
    print(recall_fact.invoke({"key": "breed", "profile": "dog"}))
    print(recall_fact.invoke({"key": "user_name", "profile": "user"})) # Should be there from migration
    
    print("\n--- Forgetting Fact ---")
    print(forget_fact.invoke({"key": "hobby", "profile": "mom"}))
    
    print("\n--- Final Memory List ---")
    print(list_all_memories.invoke({}))

if __name__ == "__main__":
    test_memory()
