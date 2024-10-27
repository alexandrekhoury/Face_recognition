import cv2
import time
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
from scipy.spatial.distance import euclidean
import os
import sqlite3

def get_client_info(embedding_path):
    """Retrieve the image path and name associated with a given embedding path."""
    cursor.execute("SELECT image_path, name FROM clients WHERE embedding_path = ?", (embedding_path,))
    result = cursor.fetchone()
    return (result[0], result[1]) if result else (None, None)

def is_match(current_embedding, saved_embedding):
    """Compare current embedding to saved embedding."""
    # Flatten the embeddings to ensure they are 1-dimensional
    current_embedding = current_embedding.flatten()
    saved_embedding = saved_embedding.flatten()
    
    distance = euclidean(current_embedding, saved_embedding)
    return distance < MATCH_THRESHOLD

def create_embedding(frame):
    """Detect faces, create embeddings, and compare with each saved embedding in the directory."""
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            width, height = x2 - x1, y2 - y1

            # Ensure the detected face meets the minimum size
            if width >= MIN_FACE_SIZE[0] and height >= MIN_FACE_SIZE[1]:
                # Extract and resize the face region if needed
                face = frame[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (160, 160))  # Resize to 160x160, input size expected by FaceNet

                # Convert to tensor and normalize
                face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
                face_tensor = face_tensor.to(torch.float32)

                # Generate the embedding for the detected face
                try:
                    embedding = model(face_tensor).detach().numpy()

                    # Loop through each file in the face_embeddings directory to check for a match
                    for filename in os.listdir(embedding_dir):
                        file_path = os.path.join(embedding_dir, filename)

                        # Load the saved embedding
                        saved_embedding = np.load(file_path)

                        # Check if the current embedding matches the saved embedding
                        if is_match(embedding, saved_embedding):
                            image_path, client_name = get_client_info(file_path)
                            print(f"Match found with {filename}! Stopping the loop.")
                            return True, image_path, client_name  # Indicate that a match was found
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                except RuntimeError as e:
                    print(f"Error generating embedding: {e}")
            else:
                print("Face too small, skipping...")

    # Display the resulting frame with rectangles around detected faces
    cv2.imshow('Face Detection with Rectangle', frame)
    return False, None, None  # Indicate that no match was found
       # Set a target FPS (frames per second)

# Connect to the database
conn = sqlite3.connect('clients.db')
cursor = conn.cursor()

MATCH_THRESHOLD = 0.6 # Facial recognition
embedding_dir = os.path.join("client_data", "embeddings")

mtcnn = MTCNN(keep_all=True)  # Face detection
model = InceptionResnetV1(pretrained='vggface2').eval()  # Face recognition
MIN_FACE_SIZE = (80, 80)  # Adjust this value as needed


target_fps = 10
frame_time = 1.0 / target_fps
# Initialize the face detector with Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(4)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

client_image_path = None
client_name = None

while True:
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    match_found,client_image_path,client_name = create_embedding(frame)
     # Stop the loop if a match is found
    if match_found:
        break
 
    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Control frame rate
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_time:
        time.sleep(frame_time - elapsed_time)
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Display client image and name if a match was found
if client_image_path:
    client_image = cv2.imread(client_image_path)
    cv2.imshow("Client Image", client_image)
    print(f"Client Name: {client_name}")
    print("Press any key to close the client image.")
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()
# Close the database connection
conn.close()
