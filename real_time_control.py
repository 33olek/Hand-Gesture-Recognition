import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui

# Wczytanie modelu i etykiet
model = tf.keras.models.load_model("models/gesture_model.h5")
labels = np.load("models/labels.npy")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Funkcje sterowania
def do_action(gesture):
    if gesture == "fist":
        pyautogui.press("space")  # np. pauza/odtwarzanie
    elif gesture == "like":
        pyautogui.press("right")  # np. następny slajd
    elif gesture == "open":
        pyautogui.press("left")   # poprzedni slajd
    elif gesture == "ok":
        pyautogui.press("up")     # np. zwiększ głośność

# Kamera
cap = cv2.VideoCapture(0)
print("🎥 Uruchomiono rozpoznawanie gestów (Q = wyjście)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in handLms.landmark:
                landmarks += [lm.x, lm.y, lm.z]
            pred = model.predict(np.array([landmarks]))
            idx = np.argmax(pred)
            gesture = labels[idx]
            cv2.putText(frame, f"{gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            do_action(gesture)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
