import os
import datetime
import time
import pyrebase
from dotenv import load_dotenv
import cv2
import numpy as np
import requests

# Load environment variables from .env file
load_dotenv()

# Your Firebase Web Config from environment variables
config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL", "")
}

# Initialize Firebase
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

# Create a local storage folder
save_folder = r"G:\My Drive\University Files\4th Semester\Machine Learning\Final Project\Firebase Upload\photos"
os.makedirs(save_folder, exist_ok=True)

# Take a photo
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = f"photo_{timestamp}.jpg"
filepath = os.path.join(save_folder, filename)

# Capture photo using OpenCV
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not access the camera.")

ret, frame = camera.read()
if not ret:
    raise RuntimeError("Failed to capture image from the camera.")

# Save the captured photo
cv2.imwrite(filepath, frame)
camera.release()

# Check if the file was created successfully
if not os.path.exists(filepath):
    raise FileNotFoundError(f"File not found: {filepath}")

print(f"Photo saved at {filepath}")

# --- Upload to Firebase Storage ---
firebase_path = f"photos/{filename}"
storage.child(firebase_path).put(filepath)

print(f"Photo {filename} successfully uploaded to Firebase Storage ({firebase_path})")

# Generate the download URL for the uploaded file
file_url = "https://firebasestorage.googleapis.com/v0/b/plantesa-c3798.firebasestorage.app/o/photos%2F20250603_231400.jpg?alt=media&token=63cadef3-cf16-4e2a-bc96-5d99e895df68"
print(f"File can be accessed at: {file_url}")

# Ensure the file URL is valid
if not file_url:
    raise ValueError("File URL is empty. Please check the upload process.")

# Download the image from Firebase
response = requests.get(file_url)
if response.status_code == 200:
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
else:
    raise RuntimeError(f"Failed to download the image from Firebase. Status code: {response.status_code}")

# Load the Haar cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale for face detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around detected faces and add a description
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the image with detected faces and description
import matplotlib.pyplot as plt

# Convert the image from BGR to RGB for displaying
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Ensure the image is defined by downloading it again
response = requests.get(file_url)
if response.status_code == 200:
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
else:
    raise RuntimeError(f"Failed to download the image from Firebase. Status code: {response.status_code}")

# Convert the image to HSV color space for better color segmentation
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green color in HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([85, 255, 255])

# Create a mask for green color
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Perform morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
leaf_image = image.copy()
for contour in contours:
    if cv2.contourArea(contour) > 500:  # Filter small contours
        cv2.drawContours(leaf_image, [contour], -1, (0, 255, 0), 2)

# Display the image with detected leaves
leaf_image_rgb = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2RGB)

# Apply the mask to the grayscale image to isolate the detected leaves
leaf_gray_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

# Find minimum area rectangles for the detected leaves
leaf_rotated_boxes = []
for contour in contours:
    if cv2.contourArea(contour) > 500:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        leaf_rotated_boxes.append(box)

# Draw rotated rectangles on a copy of the grayscale image
leafs_with_rotated_boxes = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
for box in leaf_rotated_boxes:
    cv2.drawContours(leafs_with_rotated_boxes, [box], 0, (0, 255, 0), 2)

# Define the range for brown color in HSV (expanded range for better sensitivity)
lower_brown = np.array([8, 40, 20])
upper_brown = np.array([25, 255, 220])

# Create a mask for brown color
brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

# Perform morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)

# Find contours in the brown mask
brown_contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image using edge detection
brown_image = image.copy()
edges = cv2.Canny(brown_mask, 50, 150)
edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in edge_contours:
    if cv2.contourArea(contour) > 300:
        cv2.drawContours(brown_image, [contour], -1, (0, 0, 255), 2)

brown_image_rgb = cv2.cvtColor(brown_image, cv2.COLOR_BGR2RGB)

# Apply the brown mask to the grayscale image to isolate the detected brown spots
brown_spots_gray_image = cv2.bitwise_and(gray_image, gray_image, mask=brown_mask)

# Draw bounding boxes around brown spots on the grayscale image
brown_boxes_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
for contour in brown_contours:
    x, y, w, h = cv2.boundingRect(contour)
    if cv2.contourArea(contour) > 300:
        cv2.rectangle(brown_boxes_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Calculate the percentage of brown spots relative to green leaves
green_pixels = np.count_nonzero(mask)
brown_pixels = np.count_nonzero(brown_mask)

if green_pixels > 0:
    brown_percentage = (brown_pixels / green_pixels) * 100
    print(f"Brown spots cover {brown_percentage:.2f}% of the green leaf area.")
else:
    print("No green leaf area detected in the image.")

if 'timestamp' not in globals():
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
if 'filename' not in globals():
    filename = f"photo_{timestamp}.jpg"

# Initialize Firebase Database and upload brown spots data
db = firebase.database()
brown_spots_data = {
    "timestamp": timestamp,
    "filename": filename,
    "brown_percentage": brown_percentage
}
db.child("brown_spots_percentage").push(brown_spots_data)
print("Brown spots percentage uploaded to Firebase.")

from tensorflow import keras
import cv2
import numpy as np

# Load the trained .keras model
model = keras.models.load_model(r"C:\Users\User\Documents\tomato_disease_detector_loss-0.2826_acc-90.00.keras")
input_size = (256, 256)
leaf_img_resized = cv2.resize(image, input_size)
leaf_img_rgb = cv2.cvtColor(leaf_img_resized, cv2.COLOR_BGR2RGB)
leaf_img_norm = leaf_img_rgb / 255.0
input_tensor = np.expand_dims(leaf_img_norm, axis=0)
predictions = model.predict(input_tensor)
predicted_class = np.argmax(predictions, axis=1)[0]
class_names = [
    'Bacterial_spot',
    'Early_blight',
    'Late_blight',
    'Leaf_Mold',
    'Septoria_leaf_spot',
    'Spider_mites Two-spotted_spider_mite',
    'Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus',
    'healthy'
]
print(f"Predicted leaf condition: {class_names[predicted_class]}")

config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL")
}
db = firebase.database()
data = {
    "timestamp": timestamp,
    "filename": filename,
    "predicted_condition": class_names[predicted_class]
}
db.child("leaf_conditions").push(data)
print("Predicted leaf condition sent to Firebase.")

# Process the image by drawing annotations
photo_with_brown = image.copy()
for contour in brown_contours:
    if cv2.contourArea(contour) > 300:
        cv2.drawContours(photo_with_brown, [contour], -1, (0, 0, 255), 2)
brown_text = f"Brown: {brown_percentage:.2f}%"
cv2.putText(photo_with_brown, brown_text, (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
condition_text = f"Condition: {class_names[predicted_class]}"
cv2.putText(photo_with_brown, condition_text, (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv2.putText(photo_with_brown, f"Timestamp: {timestamp}", (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
labeled_photo_filename = filename.replace('.jpg', '_brown_labeled.jpg')
labeled_photo_filepath = os.path.join(save_folder, labeled_photo_filename)
labeled_photo_firebase_path = f"photos/{labeled_photo_filename}"
cv2.imwrite(labeled_photo_filepath, photo_with_brown)
print(f"Labeled photo saved locally at {labeled_photo_filepath}")
storage.child(labeled_photo_firebase_path).put(labeled_photo_filepath)
print(f"Labeled photo uploaded to Firebase Storage at {labeled_photo_firebase_path}")
labeled_photo_data = {
    "timestamp": timestamp,
    "filename": labeled_photo_filename,
    "labeled_photo_url": storage.child(labeled_photo_firebase_path).get_url(None)
}
db.child("labeled_photos").push(labeled_photo_data)
print("Labeled photo link uploaded to Firebase Realtime Database.")

next_time = datetime.datetime.now() + datetime.timedelta(minutes=30)
# Prepare the data for the next estimated time
next_time_data = {
    "timestamp": timestamp,
    "next_estimated_time": next_time.strftime("%Y-%m-%d %H:%M:%S")
}

# Upload the next estimated time to Firebase Realtime Database
db.child("next_photo_estimate").push(next_time_data)

print(f"Next estimated photo time ({next_time.strftime('%Y-%m-%d %H:%M:%S')}) uploaded to Firebase.")
    
