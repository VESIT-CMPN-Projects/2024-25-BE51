import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def load_and_process_music_data(file_path):
    # Load the music DataFrame
    df = pd.read_csv(file_path)

    # Drop rows with no song names and no URI
    df = df.dropna(subset=['song_name', 'uri'])

    # Remove duplicate rows based on 'song_name' and 'uri'
    df = df.drop_duplicates(subset=['song_name', 'uri'])

    # Select relevant columns for clustering
    cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'uri', 'genre', 'song_name']
    filtered_df = df[cols]

    # Identify numerical columns
    num_cols = [i for i in filtered_df.columns if filtered_df[i].dtype != 'object']
    
    # Apply StandardScaler
    scaler = StandardScaler()
    filtered_df[num_cols] = scaler.fit_transform(filtered_df[num_cols])

    # Drop non-numeric columns for PCA and clustering
    X = filtered_df.drop(['uri', 'genre', 'song_name'], axis=1)

    # Apply PCA for dimensionality reduction (7 components)
    pca = PCA(n_components=7)
    pca_result = pca.fit_transform(filtered_df[num_cols])

    # KMeans Clustering on PCA results
    kmeans = KMeans(n_clusters=7, random_state=42)
    filtered_df['cluster'] = kmeans.fit_predict(pca_result)

    # Map clusters to moods
    mood_labels = ['Suprise', 'Disgust', 'Angry', 'Sad', 'Neutral', 'Fear', 'Happy']
    filtered_df['mood'] = filtered_df['cluster'].map(lambda x: mood_labels[x])

    return filtered_df
