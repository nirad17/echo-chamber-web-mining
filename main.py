import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import networkx as nx

# Load your dataset
df = pd.read_csv("preprocessed_twitter_sentiment_data.csv")

# Step 1: Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['message'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Step 2: Cluster Analysis based on Sentiment
df['sentiment_category'] = np.where(df['sentiment'] > 0, 'Positive', np.where(df['sentiment'] < 0, 'Negative', 'Neutral'))

# Step 3: Keyword Analysis
positive_keywords = df[df['sentiment_category'] == 'Positive']['message'].str.split(expand=True).stack().value_counts().head(10)
negative_keywords = df[df['sentiment_category'] == 'Negative']['message'].str.split(expand=True).stack().value_counts().head(10)
neutral_keywords = df[df['sentiment_category'] == 'Neutral']['message'].str.split(expand=True).stack().value_counts().head(10)

# Step 4: Network Analysis
G = nx.Graph()
for index, row in df.iterrows():
    G.add_node(row['username'], sentiment=row['sentiment'])
    if index > 0:
        G.add_edge(row['username'], df.at[index - 1, 'username'])

# Additional Network Metrics
degree_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Step 5: Echo Chamber Identification
# Conduct community detection using Louvain method
communities = nx.algorithms.community.greedy_modularity_communities(G)

# Step 6: Sentiment Evolution Over Time
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Calculate rolling average sentiment over a window of time
df['rolling_avg_sentiment'] = df['sentiment'].rolling(window=50, min_periods=1).mean()

# Visualization
plt.figure(figsize=(15, 8))

# Plot Sentiment Trends Over Time
sns.lineplot(x='timestamp', y='rolling_avg_sentiment', data=df, label='Sentiment Trend', color='skyblue')

# Plot Network Metrics
sns.lineplot(x='timestamp', y=list(degree_centrality.values()), label='Degree Centrality', color='orange', alpha=0.7)
sns.lineplot(x='timestamp', y=list(closeness_centrality.values()), label='Closeness Centrality', color='green', alpha=0.7)
sns.lineplot(x='timestamp', y=list(betweenness_centrality.values()), label='Betweenness Centrality', color='purple', alpha=0.7)

plt.title('Sentiment Trends and Network Metrics Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Score')
plt.legend(loc='upper left')
plt.show()
