import os
import openai

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from llmmodels.config import API_KEY_CONFIG
from llmmodels.config import template
from llmmodels.config import MODEL_CONFIGS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

class ConversationManager:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history",input_key="human_input")# Persistent memory for conversation history

    def switch_model(self, model_name: str):
        """Switch the current model."""
        if model_name in MODEL_CONFIGS:
            self.model_name = model_name
            self.llm = getLLMModel(model_name)
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
            response= self.conversation.predict(human_input=user_input)
            print(response)
            return response
        else:
            return "Failed to get response from model"


def getLLMModel(model:str):
    if(model.startswith("gpt-")):
        llm_chat = ChatOpenAI(model_name=model, temperature=0.7,openai_api_key=getAPIKey(model))
        return llm_chat
    elif(model.startswith("Zephyr")):
        llm = HuggingFaceEndpoint(
        repo_id=MODEL_CONFIGS[model],
        task="text-generation"
        )
        llm_chat = ChatHuggingFace(llm = llm)
        return llm_chat
    else:
        llm_chat = ChatOpenAI(model_name=model, temperature=0.7,openai_api_key=getAPIKey(model))
        return llm_chat
    

def getAPIKey(model:str):
    if(model.startswith("gpt-")):
           openai.api_key = os.getenv('OPENAI_API_KEY')
           if(openai.api_key is None):
               return API_KEY_CONFIG[model]
           else:
               return openai.api_key
