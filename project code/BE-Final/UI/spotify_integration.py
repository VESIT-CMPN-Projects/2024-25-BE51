# spotify_integration.py

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import time

class SpotifyIntegration:
    def __init__(self, token_file="../spotify_token.json"):
        self.token_file = token_file

    def load_token(self):
        try:
            with open(self.token_file, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_token(self, data):
        with open(self.token_file, "w") as file:
            json.dump(data, file, indent=4)

    def authenticate_spotify(self):
        token_data = self.load_token()
        if not token_data.get("client_id") or not token_data.get("client_secret"):
            token_data["client_id"] = input("Enter Spotify Client ID: ")
            token_data["client_secret"] = input("Enter Spotify Client Secret: ")
            token_data["redirect_uri"] = "http://127.0.0.1:8888/callback"
            self.save_token(token_data)

        auth_manager = SpotifyOAuth(
            client_id=token_data["client_id"],
            client_secret=token_data["client_secret"],
            redirect_uri=token_data["redirect_uri"],
            scope="user-modify-playback-state user-read-playback-state"
        )

        token_info = auth_manager.get_access_token(as_dict=True)
        token_info["expires_at"] = time.time() + token_info["expires_in"]
        self.save_token({**token_data, **token_info})
        return spotipy.Spotify(auth=token_info["access_token"])

    def refresh_token(self):
        token_data = self.load_token()
        if "refresh_token" not in token_data:
            print("⚠️ No refresh token found. Re-authenticating...")
            return self.authenticate_spotify()
        
        auth_manager = SpotifyOAuth(
            client_id=token_data["client_id"],
            client_secret=token_data["client_secret"],
            redirect_uri=token_data["redirect_uri"]
        )

        token_info = auth_manager.refresh_access_token(token_data["refresh_token"])
        token_info["expires_at"] = time.time() + token_info["expires_in"]
        self.save_token({**token_data, **token_info})
        return spotipy.Spotify(auth=token_info["access_token"])

    def get_spotify_client(self):
        token_data = self.load_token()
        if "expires_at" not in token_data or time.time() > token_data["expires_at"]:
            print("🔄 Token expired. Refreshing token...")
            return self.refresh_token()
        return spotipy.Spotify(auth=token_data["access_token"])

    def play_track_by_name(self, song_name):
        sp = self.get_spotify_client()
        results = sp.search(q=song_name, limit=1, type='track')
        if results["tracks"]["items"]:
            track_uri = results["tracks"]["items"][0]["uri"]
            print(f"🔍 Found song: {song_name} (URI: {track_uri})")
            devices = sp.devices()
            device_id = None
            if devices["devices"]:
                device_id = devices["devices"][0]["id"]
            if device_id:
                sp.start_playback(device_id=device_id, uris=[track_uri])
                print(f"🎵 Now playing: {song_name}")
            else:
                print("⚠️ No active device found.")
        else:
            print(f"❌ No results found for '{song_name}'.")
