import cv2
import mediapipe as mp
import os
import pandas as pd

DATASET_DIR = "dataset"
OUTPUT_FILE = "data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.1,  # BARDZO NISKA CZUŁOŚĆ
    min_tracking_confidence=0.1
)
mp_draw = mp.solutions.drawing_utils


def extract_landmarks_from_image(image_path):
    """Ekstrahuje landmarki z pojedynczego obrazu"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ❌ Nie można wczytać obrazu: {image_path}")
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            return None  # brak dłoni w obrazie

        hand = result.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        return landmarks
    except Exception as e:
        print(f"  ❌ Błąd przetwarzania {image_path}: {e}")
        return None


def load_dataset():
    """Ładuje dataset i tworzy plik CSV"""
    rows = []
    total_images = 0
    successful_images = 0
    failed_images = 0

    print("=" * 60)
    print("🔍 ROZPOCZYNAM EKSTRAKCJĘ LANDMARKÓW")
    print("=" * 60)

    # Sprawdź czy folder dataset istnieje
    if not os.path.exists(DATASET_DIR):
        print(f"❌ BŁĄD: Folder '{DATASET_DIR}' nie istnieje!")
        print(f"   Stwórz folder: mkdir {DATASET_DIR}")
        return

    # Pobierz listę folderów (gestów)
    labels = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

    if len(labels) == 0:
        print(f"❌ BŁĄD: Brak podfolderów w '{DATASET_DIR}'!")
        print(f"   Struktura powinna być:")
        print(f"   {DATASET_DIR}/")
        print(f"   ├── gest1/")
        print(f"   │   ├── img1.jpg")
        print(f"   │   └── img2.jpg")
        print(f"   └── gest2/")
        print(f"       └── ...")
        return

    print(f"📁 Znaleziono {len(labels)} gestów: {', '.join(labels)}")
    print()

    for label in labels:
        label_path = os.path.join(DATASET_DIR, label)

        # Pobierz listę obrazów
        images = [f for f in os.listdir(label_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if len(images) == 0:
            print(f"⚠️  '{label}': Brak obrazów w folderze")
            continue

        print(f"📂 Przetwarzam '{label}' ({len(images)} obrazów)...")
        label_success = 0

        for img_name in images:
            img_path = os.path.join(label_path, img_name)
            total_images += 1

            landmarks = extract_landmarks_from_image(img_path)
            if landmarks is None:
                failed_images += 1
                print(f"  ⚠️  {img_name}: Nie wykryto dłoni")
                continue

            row = landmarks + [label]
            rows.append(row)
            successful_images += 1
            label_success += 1
            print(f"  ✓ {img_name}: OK")

        print(f"  → Sukces: {label_success}/{len(images)} obrazów")
        print()

    print("=" * 60)
    print("📊 PODSUMOWANIE")
    print("=" * 60)
    print(f"Wszystkie obrazy: {total_images}")
    print(f"Wykryto dłoń:     {successful_images} ✓")
    print(f"Nie wykryto:      {failed_images} ✗")
    print()

    if len(rows) == 0:
        print("❌ BŁĄD: Nie udało się wyekstrahować żadnych danych!")
        print()
        print("💡 WSKAZÓWKI:")
        print("   1. Upewnij się, że na zdjęciach widoczna jest dłoń")
        print("   2. Dłoń powinna zajmować znaczną część obrazu")
        print("   3. Dobre oświetlenie pomoże w detekcji")
        print("   4. Format plików: JPG, PNG, BMP")
        return

    # Zapisz do CSV
    columns = [f"x{i}" for i in range(63)] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Zapisano {len(rows)} próbek do '{OUTPUT_FILE}'")
    print()
    print("📈 Rozkład danych:")
    print(df['label'].value_counts())
    print()
    print("🚀 Teraz możesz uruchomić trenowanie modelu!")
    print("   python main.py → wybierz opcję 2")
    print("=" * 60)


if __name__ == "__main__":
    load_dataset()