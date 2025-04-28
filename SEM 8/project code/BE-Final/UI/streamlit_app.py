import streamlit as st
import random
import cv2  
import pandas as pd
import tempfile
import os
from emotion_detection import EmotionDetection
from spotify_integration import SpotifyIntegration
from music_processing import load_and_process_music_data
from movie_recommendation import load_movie_dataset, get_movie_recommendations  # Import movie logic

# Paths for the model and cascade
model_path = '../model/model_new_training.h5'
cascade_path = cv2.data.haarcascades + '../Data/haarcascade_frontalface_default.xml'

# Initialize modules
emotion_detector = EmotionDetection(model_path, cascade_path)
spotify_integration = SpotifyIntegration()
music_df = load_and_process_music_data('../Data/genres_v2.csv')
movie_df = load_movie_dataset('../Data/movie_dataset.csv')

# Streamlit Layout
st.set_page_config(page_title="Emoverse - Emotion Detection", layout="centered")
st.title("🎭 Emoverse")
st.subheader("Let's capture your mood!")

# Take picture
image_data = st.camera_input("Take a picture to detect your emotion:")

if image_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_data.getvalue())
        temp_image_path = temp_file.name

    img = cv2.imread(temp_image_path)
    img, predicted_mood = emotion_detector.detect_emotion(img)

    st.image(img, channels="BGR", caption=f"Detected Mood: {predicted_mood}")

    if predicted_mood != "No face detected":
        st.success(f"🎉 Emotion detected: **{predicted_mood}**")

        # Let user choose what they want
        choice = st.radio("What would you like to do?", ["🎵 Recommend Songs", "🎬 Recommend Movies"])

        # 🎵 SONG RECOMMENDATION
        if choice == "🎵 Recommend Songs":
            if predicted_mood in music_df['mood'].values:
                filtered_by_mood = music_df[music_df['mood'] == predicted_mood]
                random_songs = random.sample(list(filtered_by_mood['song_name'].values), min(5, len(filtered_by_mood)))

                song_dropdown = st.selectbox("🎵 Choose a song to play", random_songs)

                if st.button('Play Song'):
                    spotify_integration.play_track_by_name(song_dropdown)
                    st.success(f"🎵 Now playing: {song_dropdown}")
            else:
                st.warning("No music available for this mood.")

        # 🎬 MOVIE RECOMMENDATION
        elif choice == "🎬 Recommend Movies":
            movie_recs = get_movie_recommendations(predicted_mood, movie_df)
            st.subheader("🎬 Movie Recommendations")

            if movie_recs:
                selected_movie = st.selectbox("Pick a movie to explore trailer:", 
                                              [f"{m['movie_name']} (⭐ {m['Ratings']})" for m in movie_recs])
                if st.button("Watch Trailer"):
                    query = selected_movie.split(" (⭐")[0].replace(" ", "+") + "+movie+trailer"
                    youtube_url = f"https://www.youtube.com/results?search_query={query}"
                    st.markdown(f"[Watch Trailer on YouTube]({youtube_url})", unsafe_allow_html=True)
            else:
                st.warning("No suitable movies found for this emotion.")
    else:
        st.warning("No face detected. Please try again.")

# # app.py
# import streamlit as st
# import random
# import cv2  
# import pandas as pd
# import tempfile  # Import tempfile module
# import os
# from emotion_detection import EmotionDetection
# from spotify_integration import SpotifyIntegration

# # Paths for the model and cascade
# model_path = '../model/model_new_training.h5'
# cascade_path = cv2.data.haarcascades + '../Data/haarcascade_frontalface_default.xml'

# # Initialize the emotion detection and Spotify integration
# emotion_detector = EmotionDetection(model_path, cascade_path)
# spotify_integration = SpotifyIntegration()

# # Assume the music DataFrame is loaded from somewhere
# # This should be pre-loaded for mood-based recommendations
# filtered_df_pca = pd.read_csv('../Data/genres_v2.csv')

# # Streamlit App Layout
# st.set_page_config(page_title="Emoverse - Emotion Detection", layout="centered")
# st.title("🎭 Emoverse")
# st.subheader("Let's capture your mood!")

# image_data = st.camera_input("Take a picture to detect your emotion:")

# if image_data is not None:
#      # Save the captured image to a temporary file using mkstemp
#     temp_fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
#     with os.fdopen(temp_fd, 'wb') as temp_file:
#         temp_file.write(image_data.getvalue())

#     img = cv2.imread(temp_image_path)
#     img, predicted_mood = emotion_detector.detect_emotion(img)

#     st.image(img, channels="BGR", caption=f"Detected Mood: {predicted_mood}")

#     if predicted_mood != "No face detected":
#         st.success(f"🎉 Emotion detected: **{predicted_mood}**")

#         # Song Recommendation based on detected mood
#         filtered_by_mood = filtered_df_pca[filtered_df_pca['emotion'] == predicted_mood]
#         random_songs = random.sample(list(filtered_by_mood['song_name'].values), min(5, len(filtered_by_mood)))

#         song_dropdown = st.selectbox("🎵 Choose a song to play", random_songs)

#         # Button to play the selected song
#         if st.button('Play Song'):
#             spotify_integration.play_track_by_name(song_dropdown)
#             st.success(f"🎵 Now playing: {song_dropdown}")













# # # streamlit_app.py

# # import streamlit as st
# # import cv2
# # import numpy as np
# # from keras.models import load_model
# # from PIL import Image
# # import tempfile
# # import os

# # # Load model and cascade
# # model_path = '../model/model_new_training.h5'
# # cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# # classifier = load_model(model_path)
# # face_cascade = cv2.CascadeClassifier(cascade_path)

# # emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # st.set_page_config(page_title="Emoverse - Emotion Detection", layout="centered")

# # st.title("🎭 Emoverse")
# # st.subheader("Let's capture your mood!")

# # # Camera input
# # image_data = st.camera_input("Take a picture to detect your emotion:")

# # if image_data is not None:
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
# #         temp_file.write(image_data.getvalue())
# #         temp_image_path = temp_file.name

# #     img = cv2.imread(temp_image_path)
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# #     predicted_mood = "No face detected"

# #     for (x, y, w, h) in faces:
# #         face_gray = gray[y:y+h, x:x+w]
# #         face_resized = cv2.resize(face_gray, (48, 48))
# #         face_array = np.expand_dims(face_resized, axis=0)
# #         face_array = np.expand_dims(face_array, axis=-1)
# #         face_array = face_array.astype('float32') / 255.0

# #         prediction = classifier.predict(face_array)
# #         max_index = np.argmax(prediction[0])
# #         predicted_mood = emotion_labels[max_index]

# #         # Draw rectangle and label
# #         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
# #         cv2.putText(img, predicted_mood, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# #     st.image(img, channels="BGR", caption=f"Detected Mood: {predicted_mood}")

# #     if predicted_mood != "No face detected":
# #         st.success(f"🎉 Emotion detected: **{predicted_mood}**")
# #         choice = st.radio("What would you like?", ("🎵 Recommend a song", "🎬 Recommend a movie"))
# #         st.session_state['emotion'] = predicted_mood
# #         st.session_state['choice'] = choice
