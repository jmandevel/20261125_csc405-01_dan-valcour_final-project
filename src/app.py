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

def message(who, what):
    with sl.chat_message(who):
        sl.write(what)
    messages.append((who, what))

def ask_ai(prompt) -> str:
    message("user", prompt)
    response = invoke_cached(messages)
    message("assistant", response)
    return response

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

sl.header("Data Cleaning and Processing")

sl.subheader("Missing Values")

sl.write("We should start this project by cleaning the dataset. First, we should look for missing values.")

missing_values = data.isna().sum().rename("Missing values")
with sl.expander("Missing Values Before Cleaning"):
    sl.write(missing_values.to_frame())

with sl.expander("AI Advice"):
    ask_ai(f"Here are the missing counts by column:\n{missing_values.to_string()}\nI plan to keep rows missing artist, album, or track names because my analysis uses track IDs, genres and musical measurements. Is this a good choice? Which missing fields would change your advice?")

sl.write("There appear to be missing values for artists, album_name, and track_name attributes. These attributes are not relevant to our data analysis, so we can ignore this.")

sl.subheader("Clustering Data")
sl.write("There are a a lot of genres in the datset. Let's look at all of them.")

genres = sorted(data["track_genre"].unique())
sl.write(f"The dataset contains {len(genres):,} genres.")
with sl.expander("List of Genres"):
    sl.dataframe(pd.DataFrame({"Genre": genres}), height=300, hide_index=True)

sl.write("It will be difficult to analyze this many genres. Instead of performing our analysis per genre, let's do it on groups of genres in clusters based on similarity")

with sl.expander("AI Advice"):
    ask_ai(f"Here is a list of genres in my dataset: [{", ".join(genres)}]. I want to perform analysis on them, but there are too many. I want to group them into clusters that are similar, so later I can determine if population and energy are correlated. There are other features too, including acousticness, danceability, instrumentalness, tempo, and valence. How can I group the genres in clusters best for my later analysis without ruining the usefulness of the analysis?")
    ask_ai("I will use K-Means. What features should I use?")
    ask_ai("Why use energy if I am measuring the correlation of it later? Isn't that bad?")
