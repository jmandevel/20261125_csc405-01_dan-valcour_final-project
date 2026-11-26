from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as sl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

cluster_count = 5
clustering_features = ["acousticness", "danceability", "instrumentalness", "tempo", "valence"]
feature_names = ", ".join(clustering_features[:-1])
feature_names += f", and {clustering_features[-1]}"

sl.write(f"Let's create {cluster_count} clusters. We can decide on which genres should be in which cluster by using K-Means. We can optimize the centroids of the K-Means around the features {feature_names}. Energy and popularity are not used in this analysis because they will be compared later.")

genre_averages = data.groupby("track_genre")[clustering_features].mean()

with sl.expander("AI Advice"):
    ask_ai(f"I have average for all genres around the clustering features {feature_names} in the Pandas DataFrame genre_averages. Now, I want to do K-Means using sci-kit. Generate me code to do this.")

scaler = StandardScaler()
scaled_averages = scaler.fit_transform(genre_averages)

kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
genre_averages["Cluster"] = kmeans.fit_predict(scaled_averages) + 1

cluster_summary = genre_averages.groupby("Cluster")[clustering_features].mean()
cluster_summary["Genres"] = genre_averages.groupby("Cluster").size()

sl.write("I generated the clusters using Sci-Kit to perform the K-Means algorithm using Sci-Kit.")

sl.write("Now that the clusters are created, I want to name them based on what genres they contain.")

cluster_names = []
with sl.expander("AI: Name the Clusters"):
    cluster_genre_name_lists = []
    for cluster in range(1, cluster_count + 1):
        cluster_genres = genre_averages[genre_averages["Cluster"] == cluster]
        cluster_genre_name_lists.append(f"[{", ".join(cluster_genres.index)}]")
    cluster_genre_names = f"[{", ".join(cluster_genre_name_lists)}]"
    clusters = ask_ai(f"I want to name the kmeans cluster result. return in comma seperated list of strings with nothing else, with nothing else in your response. These are the genres per cluster: {cluster_genre_names}. Make sure each cluster has a unique name.")
    cluster_names = clusters.split(",")
    for cluster in range(cluster_count):
        cluster_names[cluster] = cluster_names[cluster].strip()

cluster_name_text = ", ".join(cluster_names[:-1])
cluster_name_text += f", and {cluster_names[-1]}"

sl.write(f"I used the AI in order to come up with names for each cluster based on the genres they contain. The names are {cluster_name_text}.")

with sl.expander("AI Advice"):
    ask_ai(f"Generate me code to output tables with a list of the genres in each cluster, each in a streamlit expander.")

for cluster in range(1, cluster_count + 1):
    cluster_genres = genre_averages[genre_averages["Cluster"] == cluster]
    with sl.expander(f"Cluster {cluster}: {cluster_names[cluster - 1]} ({len(cluster_genres)} genres)"):
        cluster_table = cluster_genres[clustering_features].round(2)
        sl.dataframe(cluster_table)

with sl.expander("AI Advice"):
    ask_ai(f"Generate me code to output a 3d scatterplot to how clusters are related to features acousticness, danceability, and insturmentalness.")

with sl.expander("K-Means Result"):
    sl.write(cluster_summary.round(2))
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for cluster in range(1, cluster_count + 1):
        cluster_genres = genre_averages[genre_averages["Cluster"] == cluster]
        ax.scatter(
            cluster_genres[clustering_features[0]],
            cluster_genres[clustering_features[1]],
            cluster_genres[clustering_features[2]],
            label=f"Cluster {cluster}",
            cmap="tab10",
            alpha=0.8,
            edgecolor='k'
        )
    ax.set_xlabel(f"Average {clustering_features[0].capitalize()}")
    ax.set_ylabel(f"Average {clustering_features[1].capitalize()}")
    ax.set_zlabel(f"Average {clustering_features[2].capitalize()}")
    ax.set_title("Scatter Plot of Genre Clusters")
    ax.legend()
    sl.pyplot(fig)

with sl.expander("AI Advice"):
    ask_ai("I finished the K-Means calucluation, but there is something that I am worried about. Songs can exist in more than one genre, so there could be multiple songs in the same cluster. Is this bad, and should I remove duplicates per cluster?")
    ask_ai("Should I remove duplicates across all clusters, or does this not matter.")
    ask_ai("Generate me code to remove duplicates using drop_duplicates. I also want to make a stacked horizontal bar chart showing amount of songs per genre and percent of duplicates.")

sl.write("It is possible for a song to exist in multiple genres. We only want a song to exist once per cluster. We need to remove songs with the same track_id in each cluster.")

genre_clusters = genre_averages[["Cluster"]].reset_index()
cluster_tracks = data.merge(genre_clusters, on="track_genre")
song_counts_before = cluster_tracks.groupby("Cluster").size()
cluster_tracks = cluster_tracks.drop_duplicates(subset=["track_id", "Cluster"], keep='first')
song_counts_after = cluster_tracks.groupby("Cluster").size()
duplicates_removed = song_counts_before - song_counts_after

percent_removed = 100 * duplicates_removed / song_counts_before

with sl.expander("Song Entries and Duplicates by Cluster"):
    sl.write(f"Removed {duplicates_removed.sum():,} duplicate song entries in total.")
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    ax.bar(song_counts_after.index, song_counts_after, label="Unique tracks")
    bars = ax.bar(
        duplicates_removed.index,
        duplicates_removed,
        bottom=song_counts_after,
        label="Duplicate entries removed",
    )
    labels = [
        f"{percent:.1f}% removed"
        for count, percent in zip(song_counts_before, percent_removed)
    ]
    ax.bar_label(bars, labels=labels)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Song entries")
    ax.set_xticks(song_counts_before.index)
    ax.legend()
    sl.pyplot(fig)

sl.write("Now, we can analyze the clusters we created.")

with sl.expander("AI Advice"):
    ask_ai("I want to analyze the clusters with a chart of 5 number summary including standard deviation and variance. I also want box and whisker plots for the features we will analyze. I want one set for population across all 5 clusters and another set for energy.")

for column in ["energy", "popularity"]:
    with sl.expander(f"{column.capitalize()} by Cluster"):
        grouped = cluster_tracks.groupby("Cluster")[column]
        summary = grouped.describe()
        summary["var"] = grouped.var()
        sl.write(summary[["count", "min", "25%", "50%", "75%", "max", "std", "var"]].round(2))
        fig = Figure(figsize=(10, 4))
        ax = fig.subplots()
        sns.boxplot(data=cluster_tracks, x="Cluster", y=f"{column}", ax=ax)
        ax.set_ylabel(column.capitalize())
        fig.tight_layout()
        sl.pyplot(fig)
    
with sl.expander("AI Advice"):
    ask_ai("I just analyzed the energy and popularity ranges. I see a lot of outliers. Do these matter for the correlation analysis?")

sl.write("I see a lot of outliers, but I will not worry about them because they are valid data points. Removing them would make our analysis less accurate.")

with sl.expander("AI Advice"):
    ask_ai("I want to make scatterplots for each cluster with popularity on y-axis and energy on x-axis. Make me some code that does this. I want each in a seperate streamlit expander per cluster.")

for cluster in range(1, cluster_count + 1):
    with sl.expander(f"Energy and Popularity for Cluster {cluster}: {cluster_names[cluster - 1]}"):
        tracks = cluster_tracks[cluster_tracks["Cluster"] == cluster]
        track_numbers = range(1, len(tracks) + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=tracks, x='energy', y='popularity', ax=ax, alpha=0.6)
        ax.set_xlabel('Energy')
        ax.set_ylabel('Popularity')
        sl.pyplot(fig)
