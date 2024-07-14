import os
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, jsonify
import threading
import joblib
import time

app = Flask(__name__)

# Load the trained model
model_path = 'hand_shape_model.pkl'
clf = joblib.load(model_path, mmap_mode='r')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Global variable to store the detection result
detection_result = "Not Heart"

def extract_landmarks_from_frame(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
    
    while len(landmarks) < 42:
        landmarks.append([0.0, 0.0, 0.0])
    
    landmarks = np.array(landmarks).flatten()
    return landmarks

def detect_heart_shape():
    global detection_result
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        landmarks = extract_landmarks_from_frame(frame)

        if landmarks.size == 126:
            landmarks = landmarks.reshape(1, -1)
            prediction = clf.predict(landmarks)

            if prediction == 1:
                detection_result = "Heart"
            else:
                detection_result = "Not Heart"
        time.sleep(0.1)
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_detection_result', methods=['GET'])
def get_detection_result():
    return jsonify({'result': detection_result})

if __name__ == '__main__':
    # Start the background detection thread
    detection_thread = threading.Thread(target=detect_heart_shape, daemon=True)
    detection_thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
