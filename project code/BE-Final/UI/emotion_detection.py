import cv2
import numpy as np
from keras.models import load_model

class EmotionDetection:
    def __init__(self, model_path, cascade_path):
        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def detect_emotion(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        predicted_mood = "No face detected"

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_gray, (48, 48))
            face_array = np.expand_dims(face_resized, axis=0)
            face_array = np.expand_dims(face_array, axis=-1)
            face_array = face_array.astype('float32') / 255.0

            prediction = self.model.predict(face_array)
            max_index = np.argmax(prediction[0])
            predicted_mood = self.emotion_labels[max_index]

            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(img, predicted_mood, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return img, predicted_mood
