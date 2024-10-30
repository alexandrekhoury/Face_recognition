import cv2
import time
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
from scipy.spatial.distance import euclidean
import os
import sqlite3
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk

# Database and model setup
def get_client_info(embedding_path):
    """Retrieve the image path and name associated with a given embedding path."""
    cursor.execute("SELECT image_path, name FROM clients WHERE embedding_path = ?", (embedding_path,))
    result = cursor.fetchone()
    return (result[0], result[1]) if result else (None, None)

def create_embedding(frame):
    """Detect faces, create embeddings, and find the most likely match by minimum distance."""
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            width, height = x2 - x1, y2 - y1

            if width >= MIN_FACE_SIZE[0] and height >= MIN_FACE_SIZE[1]:
                
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (160, 160))
                face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
                face_tensor = face_tensor.to(torch.float32)

                try:
                    embedding = model(face_tensor).detach().numpy()
                    min_distance = float("inf")
                    closest_client = None

                    for filename in os.listdir(embedding_dir):
                        file_path = os.path.join(embedding_dir, filename)
                        saved_embedding = np.load(file_path)
                        distance = euclidean(embedding.flatten(), saved_embedding.flatten())
                        if distance < min_distance:
                            min_distance = distance
                            closest_client = file_path

                    if min_distance < MATCH_THRESHOLD:
                        image_path, client_name = get_client_info(closest_client)
                        return True, image_path, client_name

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                except RuntimeError as e:
                    print(f"Error generating embedding: {e}")
            else:
                print("Face too small, skipping...")

    return False, None, None

# Set up database and model
conn = sqlite3.connect('clients.db')
cursor = conn.cursor()
MATCH_THRESHOLD = 0.8
embedding_dir = os.path.join("client_data", "embeddings")
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()
MIN_FACE_SIZE = (80, 80)

# Tkinter window setup
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("1600x1200")  # Double the window size

        # Frame for video display and controls
        self.video_label = Label(root)
        self.video_label.pack(pady=10)

        # Client info display
        self.info_label = Label(root, text="Client Information: Not Detected", font=("Arial", 14))
        self.info_label.pack(pady=10)

        # Buttons for camera controls
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start Camera", command=self.start_camera, width=15)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop Camera", command=self.stop_camera, width=15)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.quit_button = tk.Button(button_frame, text="Quit", command=root.quit, width=15)
        self.quit_button.grid(row=0, column=2, padx=5)

        self.cap = None
        self.update_frame_id = None  # To store the ID of the scheduled update_frame call
        self.running = False     
        self.match_found = False
        self.client_image_path = None
        self.client_name = None
        self.confirmed_clients = set()
        self.paused = False  # Flag to indicate if video feed is paused

    
    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open video stream.")
                self.cap = None
                return
            self.paused = False
            self.running = True
            self.update_frame()
        else:
            print("Camera is already running.")
        
    def stop_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.update_frame_id is not None:
                try:
                    self.root.after_cancel(self.update_frame_id)
                except Exception as e:
                    print(f"Exception in after_cancel: {e}")
                self.update_frame_id = None
            cv2.destroyAllWindows()
            self.video_label.config(image="")
            self.paused = True
        else:
            print("Camera is not running.")
    def show_client_image(self):
        """Display the detected client's image before prompting for confirmation."""
        if self.client_image_path:
            self.paused = True  # Pause the video feed
            client_image = cv2.imread(self.client_image_path)
            client_image = cv2.resize(client_image, (640, 480))
            client_image = cv2.cvtColor(client_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(client_image)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display the image in a new Toplevel window
            self.top_image_window = tk.Toplevel(self.root)
            self.top_image_window.title(f"Client: {self.client_name}")
            label = Label(self.top_image_window, image=imgtk)
            label.image = imgtk  # Keep a reference
            label.pack()

            # Add confirm and cancel buttons
            button_frame = tk.Frame(self.top_image_window)
            button_frame.pack(pady=10)

            confirm_button = tk.Button(button_frame, text="Confirm", command=self.client_confirmed)
            confirm_button.pack(side=tk.LEFT, padx=5)

            cancel_button = tk.Button(button_frame, text="Cancel", command=self.client_cancelled)
            cancel_button.pack(side=tk.LEFT, padx=5)

            # Optionally, set the window to stay on top
            self.top_image_window.attributes('-topmost', True)
    

    def client_confirmed(self):
        messagebox.showinfo("Client Login", f"{self.client_name} logged in successfully!")
        self.confirmed_clients.add(self.client_name)
        self.close_client_window()

    def client_cancelled(self):
        self.info_label.config(text="Client Information: Not Detected")
        self.close_client_window()

    def close_client_window(self):
        self.top_image_window.destroy()
        self.match_found = False
        self.paused = False  # Resume video feed

    
    def update_frame(self):

        if not self.running:
            self.update_frame_id = None
            return
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize frame to fit Tkinter window size
                frame_display = cv2.resize(frame, (1280, 960))
                frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_display)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                if not self.paused:
                    match_found, client_image_path, client_name = create_embedding(frame)
                    if match_found:
                        if client_name not in self.confirmed_clients:
                            self.match_found = True
                            self.client_image_path = client_image_path
                            self.client_name = client_name
                            self.info_label.config(text=f"Match found: {self.client_name}")
                            self.paused = True  # Set paused to True after starting the confirmation process
                            self.show_client_image()  # Show the client's image before confirming
                
                        else:
                            print(f"Client {client_name} already confirmed, skipping.")
                else:
                    # When paused, you might still want to display a message or handle other tasks
                    pass
        # Reschedule update_frame if the camera is still running
        if self.running:
            self.update_frame_id = self.root.after(10, self.update_frame)
        else:
            self.update_frame_id = None

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

    # Close the database connection
    conn.close()
