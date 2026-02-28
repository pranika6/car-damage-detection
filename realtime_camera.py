import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("car_damage_model.h5")

IMG_SIZE = 224
classes = ['major', 'minor', 'moderate']  # Make sure order matches training

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for model
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])

    # Predict
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    label = f"{classes[class_index]} ({confidence:.2f})"

    # Display result on frame
    cv2.putText(frame, label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Car Damage Detection - Real Time", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
