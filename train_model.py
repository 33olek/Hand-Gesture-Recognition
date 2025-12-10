import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Uniwersalne importy Keras - działają z różnymi wersjami TF
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
except ImportError:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

DATA_FILE = "data.csv"
MODEL_FILE = "model.h5"
LABELS_FILE = "labels.npy"

print("=" * 60)
print("🤖 TRENOWANIE MODELU")
print("=" * 60)

# Wczytaj dane
print("📂 Wczytuję dane z", DATA_FILE)
df = pd.read_csv(DATA_FILE)
print(f"✓ Wczytano {len(df)} próbek")

X = df.drop("label", axis=1).values
y = df["label"].values

# Enkoduj etykiety
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"\n📊 Znalezione gesty: {list(encoder.classes_)}")
print(f"   Liczba klas: {len(encoder.classes_)}")

# Zapisz etykiety
np.save(LABELS_FILE, encoder.classes_)
print(f"💾 Zapisano etykiety do {LABELS_FILE}")

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, shuffle=True, random_state=42
)

print(f"\n📈 Podział danych:")
print(f"   Trening: {len(X_train)} próbek")
print(f"   Test:    {len(X_test)} próbek")

# Buduj model
print("\n🏗️  Buduję model...")
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# Trenuj model
print("\n🚀 Rozpoczynam trenowanie...")
print("=" * 60)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# Oceń model
print("\n" + "=" * 60)
print("📊 WYNIK NA DANYCH TESTOWYCH")
print("=" * 60)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Dokładność: {test_acc*100:.2f}%")
print(f"Strata:     {test_loss:.4f}")

# Zapisz model
model.save(MODEL_FILE)
print("\n" + "=" * 60)
print(f"✅ Model zapisany jako {MODEL_FILE}")
print(f"✅ Etykiety zapisane jako {LABELS_FILE}")
print("=" * 60)
print("\n🎉 Trenowanie zakończone pomyślnie!")
print("\n🚀 Następny krok - uruchom aplikację webową:")
print("   python app.py")
print("   Następnie otwórz: http://localhost:5000")
print("=" * 60)