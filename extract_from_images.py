import cv2
import mediapipe as mp
import os
import pandas as pd

DATASET_DIR = "dataset"
OUTPUT_FILE = "data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def extract_landmarks_from_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None  # brak dłoni w obrazie

    hand = result.multi_hand_landmarks[0]
    landmarks = []

    for lm in hand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return landmarks


def load_dataset():
    rows = []

    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)

        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            print(f"Przetwarzanie: {img_path}")

            landmarks = extract_landmarks_from_image(img_path)
            if landmarks is None:
                print("❌ Brak dłoni – pomijam")
                continue

            row = landmarks + [label]
            rows.append(row)

    columns = [f"x{i}" for i in range(63)] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✔ Zapisano dane do {OUTPUT_FILE}")


if __name__ == "__main__":
    load_dataset()
