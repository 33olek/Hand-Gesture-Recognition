print("=== Hand Gesture Recognition (Images) ===")
print("1️⃣ Ekstrakcja landmarków z dataset/")
print("2️⃣ Trening modelu")
print("3️⃣ Rozpoznanie gestu ze zdjęcia")

choice = input("Wybierz opcję [1/2/3]: ")

if choice == "1":
    import extract_from_images
elif choice == "2":
    import train_model
elif choice == "3":
    import predict_from_image
else:
    print("Niepoprawny wybór")
