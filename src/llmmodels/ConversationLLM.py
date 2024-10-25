import os
import openai
import pickle
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from getpass import getpass
#from google.colab import userdata
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
    
MODEL_CONFIGS = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4": "gpt-4"
}



template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)


# def getLLMResponse(query, model):
    
# # api_key = userdata.get('OPENAI_KEY')           # <-- change this as per your secret's name
#     os.environ['OPENAI_API_KEY'] = api_key
#     openai.api_key = os.getenv('OPENAI_API_KEY')
#     print(openai.api_key)
#     llm = ChatOpenAI(model_name=model, temperature=0)
#     output = llm.invoke(query)
#     print("output:"+output)
#     return query

class ConversationManager:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history",input_key="human_input")# Persistent memory for conversation history

    def switch_model(self, model_name: str):
        """Switch the current model."""
        if model_name in MODEL_CONFIGS:
            self.model_name = model_name
            #self.llm = OpenAI(model_name=model_name, openai_api_key=API_KEY_CONFIG[model_name])
            self.llm = OpenAI(model_name=model_name)
            self.conversation = LLMChain(llm=self.llm, memory=self.memory, verbose=True,prompt=prompt)
            print(f"Switched to model: {model_name}")
            return "success"
        else:
            print(f"Model {model_name} not available. Available models: {', '.join(MODEL_CONFIGS.keys())}")
            return "failed"
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
        print("Conversation memory cleared.")
    
    def ask(self, user_input: str, model: str) -> str:
        """Process user input and get a response from the model."""
        if(self.switch_model(model) =="success"):
            return self.conversation.predict(human_input=user_input)
        else:
            "Failed to get response from model"



