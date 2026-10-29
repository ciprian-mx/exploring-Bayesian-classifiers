import pandas as pd
import os

def data_loader(filepath="Dataset/final_viral_songs.csv"):
#def data_loader(filepath="Dataset/cleaned_viral_songs.csv"):

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}. Make sure you are running from the root directory.")

    df = pd.read_csv(filepath)
    
    selected_features = ['danceability', 'acousticness', 'energy', 'valence']
    #selected_features = ['danceability', 'acousticness', 'energy', 'valence', 'instrumentalness', 'tempo_bpm', 'speechiness', 'duration_sec', 'liveness', 'loudness_db', 'release_year']
    target_column = 'viral'
    
    x = df[selected_features]
    y = df[target_column]
    
    return x, y