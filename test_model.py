import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL = "best_chili_model.keras"

CLASSES = [
    "Bacterial_Spot",
    "Cercospora_Leaf_Spot",
    "Curl_Virus",
    "Healthy_Leaf"
]

model = load_model(MODEL, compile=False)

print("MODEL:", model.input_shape, model.output_shape)
print()

while True:

    path = input("Enter image path (or q to quit): ").strip()

    if path.lower() == "q":
        break

    if not os.path.exists(path):
        print("Image not found")
        print()
        continue

    image = cv2.imread(path)

    if image is None:
        print("Could not read image")
        print()
        continue

    image = cv2.resize(image, (224, 224))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    print()
    print("--------------------------------")

    for i in range(len(CLASSES)):
        print(
            CLASSES[i],
            "->",
            f"{prediction[i] * 100:.2f}%"
        )

    index = int(np.argmax(prediction))

    print("--------------------------------")
    print("FINAL:", CLASSES[index])
    print(
        "CONFIDENCE:",
        f"{prediction[index] * 100:.2f}%"
    )
    print("--------------------------------")
    print()
