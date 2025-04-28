import pandas as pd
from ast import literal_eval

# Load and process movie dataset
def load_movie_dataset(file_path):
    df = pd.read_csv(file_path)
    df["genres"] = df["genres"].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])
    return df

# Mapping dictionaries
emotion_mapping = {
    "Angry": ["anger"],
    "Disgust": ["disgust"],
    "Fear": ["fear"],
    "Happy": ["joy", "optimism"],
    "Neutral": ["anticipation", "trust", "neutral"],
    "Sad": ["sadness"],
    "Surprise": ["surprise"]
}

genre_mapping = {
    "Angry": ["Comedy", "Animation", "Family", "Adventure", "Music", "Romance"],
    "Disgust": ["Psychological", "Thriller", "Crime", "Dark Comedy"],
    "Fear": ["Horror", "Mystery", "Thriller", "Supernatural"],
    "Happy": ["Comedy", "Animation", "Family", "Adventure", "Music", "Romance"],
    "Neutral": ["Fantasy", "Mystery", "Science Fiction", "Adventure"],
    "Sad": ["Comedy", "Drama", "Romance", "Documentary", "History", "War"],
    "Surprise": ["Mystery", "Fantasy", "Adventure", "Sci-Fi"]
}

# Recommend top movies
def get_movie_recommendations(detected_emotion, df, top_n=5):
    emotions = emotion_mapping.get(detected_emotion, [])
    genres = genre_mapping.get(detected_emotion, [])

    filtered_df = df[df["emotion"].str.lower().isin(emotions)]
    filtered_df = filtered_df[filtered_df["genres"].apply(lambda g: any(genre in g for genre in genres))]

    filtered_df = filtered_df.drop_duplicates(subset="movie_name").nlargest(top_n, "Ratings")

    return filtered_df[["movie_name", "Ratings"]].to_dict(orient="records")
