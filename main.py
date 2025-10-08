import os

print("=== Hand Gesture Control ===")
print("1️⃣ Zbierz dane (save_landmarks.py)")
print("2️⃣ Wytrenuj model (train_model.py)")
print("3️⃣ Uruchom sterowanie (real_time_control.py)")

choice = input("Wybierz opcję [1/2/3]: ")

if choice == "1":
    os.system("python save_landmarks.py")
elif choice == "2":
    os.system("python train_model.py")
elif choice == "3":
    os.system("python real_time_control.py")
else:
    print("Niepoprawny wybór.")