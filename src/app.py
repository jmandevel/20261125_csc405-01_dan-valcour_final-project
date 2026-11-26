from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as sl

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

@sl.cache_data(persist="disk")
def invoke_cached(prompt):
    response = llm.invoke(prompt)
    return response.content

def ai_advice(prompt):
    with sl.expander("AI Advice"):
        with sl.chat_message("user"):
            sl.write(prompt)
        response = invoke_cached(prompt)
        with sl.chat_message("ChatGPT"):
            sl.write(response)
