import tensorflow as tf
import numpy as np
import cv2
import sys

IMG_SIZE = 224

# Load trained model
model = tf.keras.models.load_model("car_damage_model.h5")

classes = ['major', 'minor', 'moderate']  # must match folder order

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Damage Type: {classes[class_index]}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    image_path = sys.argv[1]
    predict_image(image_path)
