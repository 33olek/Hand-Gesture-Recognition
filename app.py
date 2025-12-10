from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64

# Uniwersalne importy Keras
try:
    from tensorflow.keras.models import load_model
except ImportError:
    try:
        from keras.models import load_model
    except ImportError:
        import tensorflow as tf

        load_model = tf.keras.models.load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_FILE = "model.h5"
LABELS_FILE = "labels.npy"

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

# Załaduj model i etykiety
model_loaded = False
model = None
labels = None

if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
    try:
        print(f"📂 Próbuję załadować model z: {os.path.abspath(MODEL_FILE)}")
        model = load_model(MODEL_FILE, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        labels = np.load(LABELS_FILE, allow_pickle=True)
        model_loaded = True
        print(f"✅ Model załadowany pomyślnie!")
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        model_loaded = False
else:
    print(f"⚠️ Pliki modelu nie istnieją:")
    print(f"   {MODEL_FILE}: {os.path.exists(MODEL_FILE)}")
    print(f"   {LABELS_FILE}: {os.path.exists(LABELS_FILE)}")


def extract_landmarks(img):
    """Ekstrahuje landmarki z obrazu"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None, None

    hand = result.multi_hand_landmarks[0]
    coords = []

    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    # Rysuj landmarki na obrazie
    img_with_landmarks = img.copy()
    mp_draw.draw_landmarks(img_with_landmarks, hand, mp_hands.HAND_CONNECTIONS)

    return np.array(coords).reshape(1, -1), img_with_landmarks


def predict_gesture(img):
    """Rozpoznaje gest z obrazu"""
    if not model_loaded:
        return None, None, None

    landmarks, img_with_landmarks = extract_landmarks(img)

    if landmarks is None:
        return None, None, None

    pred = model.predict(landmarks, verbose=0)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx])
    gesture = labels[idx]

    return gesture, confidence, img_with_landmarks


@app.route('/')
def index():
    """Strona główna"""
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint do rozpoznawania gestów"""
    if not model_loaded:
        return jsonify({'error': 'Model nie został wytrenowany'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nie wybrano pliku'}), 400

    filepath = None
    try:
        # Zapisz plik tymczasowo
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Wczytaj obraz
        img = cv2.imread(filepath)

        # Rozpoznaj gest
        gesture, confidence, img_with_landmarks = predict_gesture(img)

        if gesture is None:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Nie wykryto dłoni na zdjęciu'}), 400

        # Zakoduj obraz z landmarkami do base64
        _, buffer = cv2.imencode('.jpg', img_with_landmarks)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Usuń tymczasowy plik
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            'gesture': gesture,
            'confidence': round(confidence * 100, 2),
            'image': img_base64
        })

    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/gestures')
def get_gestures():
    """Zwraca listę dostępnych gestów"""
    if model_loaded and labels is not None:
        return jsonify({'gestures': labels.tolist()})
    return jsonify({'gestures': []})


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Hand Gesture Recognition")
    print("=" * 50)
    if model_loaded:
        print(f"✓ Model załadowany: {len(labels)} gestów")
        print(f"  Gesty: {', '.join(labels)}")
        if len(labels) == 1:
            print("\n⚠️  UWAGA: Masz tylko 1 gest!")
            print("   Model będzie zawsze zwracał ten sam wynik.")
            print("   Dodaj więcej gestów do dataset/ i przetrenuj ponownie.")
    else:
        print("⚠️  Model nie został wytrenowany!")
        print("   Uruchom: python train_model.py")
    print("=" * 50)
    print("📱 Otwórz: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)