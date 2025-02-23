import cv2
import numpy as np
import os

# Load trained face recognizer model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainer.yml")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Get user IDs from dataset folder
user_ids = os.listdir(r"C:\Users\User\Desktop\dataset2")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_roi)

        # Get username from dataset
        username = user_ids[label] if label < len(user_ids) else "Unknown"

        # Display recognition result
        if confidence < 70:  # Adjust confidence threshold
            text = f"{username} ({confidence:.2f})"
            color = (0, 255, 0)  # Green for recognized face
        else:
            text = "Unknown"
            color = (0, 0, 255)  # Red for unrecognized face

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
