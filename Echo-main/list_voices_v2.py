import pyttsx3

def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if "en_US" in voice.languages or "en-US" in voice.id:
            print(f"ID: {voice.id}")
            print(f"Name: {voice.name}")
            print(f"Gender: {voice.gender}")
            print("-" * 20)

if __name__ == "__main__":
    list_voices()
