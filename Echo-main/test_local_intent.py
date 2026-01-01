from main import check_local_intent
import logging

# Mock logging to avoid clutter
logging.basicConfig(level=logging.CRITICAL)

def test_local_intent():
    print("--- Testing Local Intent Logic ---")
    
    test_cases = [
        ("what is my name", "user", "name"),
        ("what's my best friend", "best friend", None), # Should trigger profile list
        ("what is my best friend's name", "best friend", "name"),
        ("what's my dog's breed", "dog", "breed"),
        ("tell me a joke", None, None),
    ]
    
    for text, expected_profile, expected_key in test_cases:
        print(f"Testing: '{text}'")
        
        result = check_local_intent(text)
        
        if expected_profile:
            if result and isinstance(result, str):
                print(f"  ✅ Match found (Result: {result[:50]}...)")
            else:
                print(f"  ❌ FAILED: Expected match for {expected_profile}, got None")
        else:
            if result is None:
                print(f"  ✅ Correctly ignored")
            else:
                print(f"  ❌ FAILED: Expected None, got result")

if __name__ == "__main__":
    test_local_intent()
