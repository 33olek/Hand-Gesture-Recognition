import cv2
import mediapipe as mp
import numpy as np
from tensorflow.python.keras.models import load_model


MODEL_FILE = "model.h5"
LABELS_FILE = "labels.npy"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

model = load_model(MODEL_FILE)
labels = np.load(LABELS_FILE)

def extract_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]
    coords = []

    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    return np.array(coords).reshape(1, -1)

def predict(img_path):
    img = cv2.imread(img_path)
    landmarks = extract_landmarks(img)

    if landmarks is None:
        print("❌ Nie wykryto dłoni!")
        return

    pred = model.predict(landmarks)
    idx = np.argmax(pred)
    print("➡ Rozpoznany gest:", labels[idx])

if __name__ == "__main__":
    path = input("Podaj ścieżkę do zdjęcia: ")
    predict(path)
