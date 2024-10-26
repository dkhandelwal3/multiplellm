API_KEY_CONFIG ={
    "gpt-3.5-turbo": "",
    "gpt-4": ""
}

MODEL_CONFIGS = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4": "gpt-4",
    "Zephyr":"HuggingFaceH4/zephyr-7b-beta"
}


template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""
