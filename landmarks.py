# save_landmarks.py
import cv2, mediapipe as mp, csv, os, time
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.6, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

labels = ["open","fist","ok","like"]  # edytuj jeśli chcesz
out_dir = "data_landmarks"
os.makedirs(out_dir, exist_ok=True)
csv_file = os.path.join(out_dir, "landmarks.csv")

if not os.path.exists(csv_file):
    with open(csv_file,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)])

cap = cv2.VideoCapture(0)
print("Nacisnij klawisz z numerem etykiety, aby zacząć nagrywać (0=open,1=fist,2=ok,3=like). q - wyjście.")
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Collect", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if chr(key).isdigit():
        idx = int(chr(key))
        if idx < 0 or idx >= len(labels):
            print("Niepoprawny numer etykiety")
            continue
        print(f"Zbieram próbki dla: {labels[idx]} (3s)...")
        t_end = time.time()+3
        samples = 0
        while time.time() < t_end:
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                zs = [lm.z for lm in hand.landmark]
                row = [labels[idx]] + xs + ys + zs
                with open(csv_file,'a',newline='') as f:
                    writer=csv.writer(f)
                    writer.writerow(row)
                samples += 1
            cv2.imshow("Collect", frame)
            cv2.waitKey(1)
        print("Zapisane:", samples, "próbek")
cap.release(); cv2.destroyAllWindows()
