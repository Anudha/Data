
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

import json

# Function to load JSON file and create a list of phrases
def load_phrases_from_json(filepath):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    return data


file_path = 'accounts_list.json'

phrases = load_phrases_from_json(file_path)

print(phrases)
print(len(phrases))  # approximately 13k phrases
# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Embed the phrases
embeddings = model.encode(phrases)

# Number of clusters (adjust this based on your data)
num_clusters = 2000  # ran for 3 , 1000, 2000 

# Cluster
clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

# Group phrases by clusters
clustered_phrases = [[] for _ in range(num_clusters)]
for phrase, cluster_id in zip(phrases, cluster_assignment):
    clustered_phrases[cluster_id].append(phrase)

# Output the clusters of similar phrases
for i, cluster in enumerate(clustered_phrases):
    print(f"Cluster {i + 1}:")
    for phrase in cluster:
        print(f"  {phrase}")
    print()

