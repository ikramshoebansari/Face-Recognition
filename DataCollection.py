import cv2
import os

# Initialize the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a dataset folder if not exists
dataset_path = r"C:\Users\User\Desktop\dataset2"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

user_id = input("Enter a unique user ID: ")
user_folder = os.path.join(dataset_path, user_id)
if not os.path.exists(user_folder):
    os.makedirs(user_folder)


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the count for image numbering
count = 0
max_samples = 150  # Limit to 150 samples

print(f"Collecting face data for user ID: {user_id}")

# Data Collection Loop (limit to 150 samples)
while count < max_samples:
    # Capture frame-by-frame
    ret, img = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        img_path = os.path.join(user_folder, f"{count}.jpg")
        cv2.imwrite(img_path, face_roi)
        count += 1

        # Draw rectangle on the original image (optional)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display image with faces detected (optional)
    cv2.imshow("Collecting Faces", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print(f"Collected {count} face images for user ID: {user_id}")
