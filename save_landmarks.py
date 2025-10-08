import cv2
import mediapipe as mp
import csv
import os
import time

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Folder do zapisu danych
out_dir = "data_landmarks"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "landmarks.csv")

# Etykiety gestów
labels = ["open", "fist", "like", "ok"]

# Tworzenie pliku CSV jeśli nie istnieje
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

# Kamera
cap = cv2.VideoCapture(0)
print("Naciśnij klawisz 0=open, 1=fist, 2=like, 3=ok, Q=wyjście.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif chr(key).isdigit():
        idx = int(chr(key))
        if idx < len(labels):
            print(f"📸 Zbieranie danych dla gestu: {labels[idx]} (3 sekundy)")
            start = time.time()
            while time.time() - start < 3:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        data = [labels[idx]]
                        for lm in handLms.landmark:
                            data += [lm.x, lm.y, lm.z]
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(data)
                        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                cv2.imshow("Collecting Data", frame)
                cv2.waitKey(1)
            print("✅ Zakończono zbieranie")

cap.release()
cv2.destroyAllWindows()
