import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
import sqlite3
import time
import sys
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "1"


# Initialize MTCNN and FaceNet model
mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define minimum face size and directory for embeddings and images
MIN_FACE_SIZE = (80, 80)
embedding_dir = "client_data/embeddings"
image_dir = "client_data/images"
os.makedirs(embedding_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Connect to the database
conn = sqlite3.connect('clients.db')
cursor = conn.cursor()

def capture_and_save_client_info(name, frame, face_region):
    """Capture the face, save embedding and image, and insert info into database."""
    # Resize the face region to the size expected by FaceNet
    face_resized = cv2.resize(face_region, (160, 160))

    # Convert to tensor and normalize
    face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
    face_tensor = face_tensor.to(torch.float32)

    # Generate embedding
    with torch.no_grad():
        embedding = model(face_tensor).detach().numpy()

    # Create unique filenames based on timestamp
    timestamp = int(time.time())
    image_filename = os.path.join(image_dir, f"{name}_{timestamp}.jpg")
    embedding_filename = os.path.join(embedding_dir, f"{name}_{timestamp}.npy")

    # Display the captured face region and ask for confirmation
    cv2.imshow("Captured Face", face_region)
    print("Are you satisfied with this image? Press 'y' for Yes, 'n' for No.")
    if cv2.waitKey(0) & 0xFF == ord('y'):
        # Save full frame as the image and embedding if confirmed
        cv2.imwrite(image_filename, frame)
        np.save(embedding_filename, embedding)

        # Insert client info into the database
        cursor.execute('''
            INSERT INTO clients (name, image_path, embedding_path)
            VALUES (?, ?, ?)
        ''', (name, image_filename, embedding_filename))
        conn.commit()

        print(f"Client '{name}' added with image '{image_filename}' and embedding '{embedding_filename}'")
        cv2.destroyWindow("Captured Face")
        return True  # Exit the loop if satisfied
    else:
        print("Retaking image...")
        cv2.destroyWindow("Captured Face")
        return False  # Continue the loop if not satisfied

# Prompt for the client's name
client_name = input("Enter the client's name: ").replace(" ", "_")

# Start video capture
cap = cv2.VideoCapture(4)
embedding_saved = False
last_capture_time = time.time()

while not embedding_saved:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Capture a frame for processing once per second
    current_time = time.time()
    if current_time - last_capture_time >= 1:
        # Detect face
        boxes, _ = mtcnn.detect(frame)
        last_capture_time = current_time  # Update last capture time

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                width, height = x2 - x1, y2 - y1

                # Only process if the face is large enough
                if width >= MIN_FACE_SIZE[0] and height >= MIN_FACE_SIZE[1]:
                    face_region = frame[y1:y2, x1:x2]  # Extract face region for confirmation
                    embedding_saved = capture_and_save_client_info(client_name, frame, face_region)

                if embedding_saved:
                    break  # Exit the loop if the user confirms the image

    # Show the live feed
    cv2.imshow('Live Face Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
conn.close()
