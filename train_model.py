import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

DATA_FILE = "data.csv"
MODEL_FILE = "model.h5"
LABELS_FILE = "labels.npy"

df = pd.read_csv(DATA_FILE)

X = df.drop("label", axis=1).values
y = df["label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

np.save(LABELS_FILE, encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, shuffle=True
)

model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dense(64, activation="relu"),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save(MODEL_FILE)
print("✔ Model zapisany jako", MODEL_FILE)
