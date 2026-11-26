from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as sl
import pandas as pd

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

@sl.cache_data(persist="disk")
def invoke_cached(messages):
    response = llm.invoke(messages)
    return response.content

messages = []

def ask_ai(prompt):
    with sl.chat_message("user"):
        sl.write(prompt)
    messages.append(prompt)
    response = invoke_cached(messages)
    with sl.chat_message("assistant"):
        sl.write(response)

@sl.cache_data(persist="disk")
def load_data():
    return pd.read_csv("data/dataset.csv", index_col=0)

sl.title("Spotify Track Energy and Popularity")
sl.markdown("This report examines how track energy is correlated to popularity across Spotify genres.")

sl.header("Data Acquisition")
sl.markdown("The [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) was collected using the Spotify API. It was created by Maharshi Pandya and released under the ODC Open Database License (ODbL).")

with sl.expander("AI Advice"):
    ask_ai("I plan to use the Kaggle Spotify Tracks Dataset (https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) to study how track energy is correlated to popularity across genres. What dataset properties and limitations should I check before using it for this question?")

data = load_data()
with sl.expander("Data Head"):
    sl.write(f"Loaded {data.shape[0]:,} rows and {data.shape[1]:,} columns. Viewing the first 5 rows.")
    sl.write(data.head())
