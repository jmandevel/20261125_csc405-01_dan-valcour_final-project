from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as sl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


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
    ask_ai("I want to make scatterplots for each cluster with popularity on y-axis and energy on x-axis. Make me some code that does this. I want each in a seperate streamlit expander per cluster. I want regression lines.")

for cluster in range(1, cluster_count + 1):
    with sl.expander(f"Energy and Popularity for Cluster {cluster}: {cluster_names[cluster - 1]}"):
        tracks = cluster_tracks[cluster_tracks["Cluster"] == cluster]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.regplot(
            x='energy',
            y='popularity',
            data=tracks,
            scatter_kws={'alpha':0.1, 's':8},
            line_kws={'color':'red'},
            ax=ax
        )
        ax.set_title(f'Popularity vs Energy - {cluster_names[cluster - 1]}')
        ax.set_xlabel('Energy')
        ax.set_ylabel('Popularity')
        sl.pyplot(fig)

sl.header("Correlation by Genre")

with sl.expander("AI Advice"):
    ask_ai("I have my clusters determined. What else should I do before doing correlation tests on energy and popularity?")
    ask_ai("If I leave outliers, will this effect normalization? Will this negatively effect a Pearson correlation")
    ask_ai("Generate me code to do Spearman correlation. cluster_count is amount of clusters, and its 1 indexed. cluster_tracks is a Pandas DataFrame containing all tracks including Cluster id.")

sl.write("Since I am leaving the outliers in my data, I will analyze my data with a Spearman correlation coefficient. A Pearson correlation would be negatively effected by so many outliers.")

corr_pvals = []
cluster_sizes = []
for cluster_id in range(1, cluster_count + 1):
    cluster_data = cluster_tracks[cluster_tracks['Cluster'] == cluster_id]
    corr, pval = spearmanr(cluster_data['energy'], cluster_data['popularity'])
    corr_pvals.append((corr, pval))
    cluster_sizes.append(len(cluster_data))

with sl.expander("AI Advice"):
    ask_ai("I want to make a bar chart showing different between correlation strengths of clusters. Names are stored in cluster_names python list of size cluster_count. cor_pvals is python list of tuples with both Spearman correlation results and pvalues. Generate code for me. It should be in al.expander(\"Cluster Correlations\"). I also want another bar graph showing strengths of the correlations.")
    ask_ai("How measure strength of Spearman correlation?")

correlations = [item[0] for item in corr_pvals]
p_values = [item[1] for item in corr_pvals]

with sl.expander("Cluster Correlations"):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cluster_names, np.abs(correlations), color='skyblue')
    ax.set_ylabel('Absolute Spearman Correlation')
    ax.set_title('Absolute Spearman Correlation Strengths by Cluster')
    ax.set_ylim(0, 1)
    ax.bar_label(bars, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    sl.pyplot(fig)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = ['green' if p <= 0.05 else 'gray' for p in p_values]
    bars2 = ax2.bar(cluster_names, correlations, color=colors)
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Spearman Correlation by Cluster (Green = p ≤ 0.05)')
    ax2.set_ylim(-1, 1)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.bar_label(bars2, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    sl.pyplot(fig2)

sl.write("Notice that energy and population have a negative correlation in the clusters \"Dance & Urban\", \"Calm & Classical\", and \"Rock & Metal\". Meanwhile, notice that that \"Folk & Acoutsic\" has a positive correlation, while \"Electronic & Techno\" has almost no correlation. All of the correlations weak, and the largest is .21 (\"Calm & Classical\").")

with sl.expander("AI Advice"):
    ask_ai("Now, I want to perform a test to see if the correlations are statistically significant between clusters. What test should I do?")
    ask_ai("Generate code to do this.")
    ask_ai("Yes, I want this integrated into my app. I want an expander showing results.")
    ask_ai("How do I describe this test? What is my Null Hypothesis? Alt Hypothesis? What do I write to streamlit?")

def fisher_z(r):
    """Convert correlation coefficient to Fisher's z."""
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlations(r1, n1, r2, n2):
    """
    Compare two independent Spearman correlations using Fisher's z-test.

    Parameters:
    - r1, r2: correlation coefficients
    - n1, n2: sample sizes used to compute r1 and r2

    Returns:
    - z_stat: z statistic for difference
    - p_value: two-tailed p-value
    """
    # Handle edge cases where r is exactly 1 or -1
    if abs(r1) == 1 or abs(r2) == 1:
        return None, None  # Cannot compute Fisher z for perfect correlations

    z1 = fisher_z(r1)
    z2 = fisher_z(r2)
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))  # two-tailed
    return z_stat, p_value

sl.write("I can perform a Fisher's z-test for comparing independent Spearman correlations. This test evaluates whether the strength of the monotonic association (Spearman correlation) between energy and popularity differs significantly between two independent clusters of songs.")
sl.write("My Null Hypothesis is the Spearman correlation between energy and popularity is the same in both clusters.")
sl.write("My Alterntive Hypothesis is the Spearman correlation between energy and popularity is different between the two clusters.")


def show_fisher_z_results(title, correlations, sizes, names):
    results = [[(None, None) for name in names] for name in names]
    rows = []

    with sl.expander(title):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                z_stat, p_val = compare_correlations(
                    correlations[i][0], sizes[i],
                    correlations[j][0], sizes[j],
                )
                results[i][j] = (z_stat, p_val)

                if p_val is None:
                    result_text = "Unavailable"
                    significant = "Unavailable"
                else:
                    result_text = f"z = {z_stat:.3f}, p = {p_val:.4g}"
                    significant = "Yes" if p_val < 0.05 else "No"

                rows.append({
                    "Cluster 1": names[i],
                    "Cluster 2": names[j],
                    "Result": result_text,
                    "Reject Null Hypothesis?": significant,
                })

        sl.table(rows)

    return results

z_test_result = show_fisher_z_results(
    "Compare Correlations Between Clusters",
    corr_pvals,
    cluster_sizes,
    cluster_names,
)

with sl.expander("AI Advice: Interpreting the Correlations"):
    prompt = "I measured the Spearman correlation between track energy and popularity within each genre cluster. I then used pairwise z tests to compare the correlations. A track counts once per cluster, but the same track can appear in more than one cluster.\n\nSpearman results by cluster:\n"
    for i in range(cluster_count):
        prompt += f"Cluster {cluster} ({cluster_names[cluster_count - 1]}): rho = {corr_pvals[i][0]:.4f}, p = {corr_pvals[i][1]:.4g}\n"
    prompt += "\nPairwise Fisher z test results:\n"
    for i in range(cluster_count):
        for j in range(cluster_count):
            if j <= i:
                continue
            prompt += f"Cluster {cluster_names[i]} vs. cluster {cluster_names[j]}: z = {z_test_result[i][j][0]:.4f}, p = {z_test_result[i][j][1]:.4g}\n"
    prompt += "\nHow should I interpret the direction and strength of the correlations? Explain the difference between testing whether one cluster's correlation is zero and testing whether two clusters' correlations differ. Are these Fisher z tests appropriate for Spearman correlations when clusters share tracks? How should I account for multiple comparisons, and what conclusions do these results support? Note that there are no duplicate tracks in the same cluster."
    ask_ai(prompt)

sl.write("We measured Spearman correlations between track energy and popularity within each genre cluster, finding mostly weak but statistically significant associations. Pairwise Fisher z-tests indicated significant differences in correlation strength between clusters. However, because some tracks appear in multiple clusters, these tests violate the assumption of independent samples, so results should be interpreted cautiously. Adjusting for multiple comparisons confirms the significance of most differences. Overall, the data suggest that the energy-popularity relationship varies across genre clusters, but further analysis accounting for overlapping tracks is needed to confirm these findings robustly.")

with sl.expander("AI Advice"):
    ask_ai("Help me to implement more robust tests.")
    ask_ai("My computer is not that powerful. What if I remove duplicates across clusters before the spearman test?")
    ask_ai("Generate code to do this. Note I still have the functions you defined earlier available. I want to remove duplicates then do the test again.")

sl.write("In order to increase the accuracy of the results, I will remove duplicates across clusters and perform the spearman correlation again. There are other tests I could perform, but my computer is not powerful enough to finish them. This is annoying because the AI previously told me not to remove the tracks because it said it will negatively effect my results.")

deduped_tracks = cluster_tracks.drop_duplicates(subset=['track_id'], keep='first')

deduped_corr_pvals = []
deduped_cluster_sizes = []

for cluster_id in range(1, cluster_count + 1):
    cluster_data = deduped_tracks[deduped_tracks["Cluster"] == cluster_id]
    cluster_data = cluster_data.dropna(subset=["energy", "popularity"])

    corr, pval = spearmanr(cluster_data["energy"], cluster_data["popularity"])
    deduped_corr_pvals.append((corr, pval))
    deduped_cluster_sizes.append(len(cluster_data))

deduped_correlations = [item[0] for item in deduped_corr_pvals]
deduped_p_values = [item[1] for item in deduped_corr_pvals]

with sl.expander("Deduped Cluster Correlations"):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cluster_names, np.abs(deduped_correlations), color='skyblue')
    ax.set_ylabel('Absolute Spearman Correlation')
    ax.set_title('Absolute Spearman Correlation Strengths by Cluster')
    ax.set_ylim(0, 1)
    ax.bar_label(bars, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    sl.pyplot(fig)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = ['green' if p <= 0.05 else 'gray' for p in deduped_p_values]
    bars2 = ax2.bar(cluster_names, deduped_correlations, color=colors)
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Spearman Correlation by Cluster (Green = p ≤ 0.05)')
    ax2.set_ylim(-1, 1)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.bar_label(bars2, fmt='%.2f')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    sl.pyplot(fig2)

sl.write(
    "I repeat the same approximate Fisher z comparisons after keeping each track ID in only one cluster. This is a sensitivity check, and the pairwise p-values are unadjusted."
)

deduped_z_test_result = show_fisher_z_results(
    "Compare Correlations Between Clusters With No Shared Tracks",
    deduped_corr_pvals,
    deduped_cluster_sizes,
    cluster_names,
)

sl.write("After removing duplicates, the differences between the correlations are still significant.")

sl.header("Popularity Predictor Model")

with sl.expander("AI Advice"):
    ask_ai("I am done with my analysis. Now, I want to shift to a different task. I want to make a machine learning model that can take in features about a track and predict if its popular or not. What should I do?")
    ask_ai("I like your example source, but I want the output in streamlit inside an expander with table showing classification report.")

sl.write("Using the Spotify Tracks Dataset, I am going to make a ML Classifier that can determine if a Spotify track is popular or not given its features. I will consider a song to be popular if its popularity percentage is greater than 70. This is a binary classification. I will create a Random Forst model, and I will split it between training and testing sets.")

data['popular'] = (data['popularity'] >= 70).astype(int)

# Select features
feature_cols = ['energy', 'danceability', 'acousticness', 'instrumentalness', 'tempo', 'valence', 'loudness']
X = data[feature_cols]
y = data['popular']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

with sl.expander("Random Forest Classification Report"):
    sl.write("Classification report for predicting track popularity (popular if popularity >= 70).")
    # Format floats nicely
    sl.dataframe(report_df.style.format({
        'precision': '{:.2f}',
        'recall': '{:.2f}',
        'f1-score': '{:.2f}',
        'support': '{:.0f}'
    }))
    sl.write(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

with sl.expander("AI Advice"):
    ask_ai(f"How good are these results?\n\n{classification_report(y_test, y_pred)}\n\nROC AUC: {roc_auc_score(y_test, y_proba)}")

sl.write("The model is good at recognizing popular tracks, but not as good at recognizing tracks that are not popular.")