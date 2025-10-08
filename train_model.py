import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Wczytanie danych
data_path = "data_landmarks/landmarks.csv"
data = pd.read_csv(data_path)

# Etykiety i dane
X = data.drop("label", axis=1).values
y = data["label"].values

# Kodowanie etykiet
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Tworzenie modelu
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(63,)),   # 21 punktów * 3 współrzędne
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie
print("🔄 Trenowanie modelu...")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Ocena
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Dokładność: {acc*100:.2f}%")

# Zapis modelu
os.makedirs("models", exist_ok=True)
model.save("models/gesture_model.h5")
np.save("models/labels.npy", le.classes_)
print("💾 Model zapisany do folderu models/")
