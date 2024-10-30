import datetime
import cv2
import time
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
from scipy.spatial.distance import euclidean
import os
import sqlite3
import tkinter as tk
from tkinter import Label, messagebox, Listbox, Scrollbar
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
                        return True, image_path, client_name, (x1, y1, x2, y2)

                except RuntimeError as e:
                    print(f"Error generating embedding: {e}")
            else:
                print("Face too small, skipping...")

    return False, None, None, None

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
        self.root.geometry("2200x1200")  # Expanded window size

        # Frame for video display
        video_frame = tk.Frame(root)
        video_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

        self.video_label = Label(video_frame)
        self.video_label.pack()

        # Frame for client info and buttons
        info_frame = tk.Frame(root)
        info_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Client info label
        self.info_label = Label(info_frame, text="Client Information: Not Detected", font=("Arial", 14))
        self.info_label.pack(pady=10)

        # Buttons for camera controls
        button_frame = tk.Frame(info_frame)
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start Camera", command=self.start_camera, width=15)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop Camera", command=self.stop_camera, width=15)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.quit_button = tk.Button(button_frame, text="Quit", command=root.quit, width=15)
        self.quit_button.grid(row=0, column=2, padx=5)

        # Frame for confirmed clients list
        confirmed_frame = tk.Frame(root)
        confirmed_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")

        confirmed_label = tk.Label(confirmed_frame, text="Confirmed Clients:", font=("Arial", 14))
        confirmed_label.pack()

        # Scrollbar for the listbox
        scrollbar = Scrollbar(confirmed_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Listbox to display confirmed clients
        self.confirmed_listbox = Listbox(confirmed_frame, width=50, height=20, yscrollcommand=scrollbar.set)
        self.confirmed_listbox.pack(side=tk.LEFT, fill=tk.BOTH)

        scrollbar.config(command=self.confirmed_listbox.yview)

        self.cap = None
        self.update_frame_id = None  # To store the ID of the scheduled update_frame call
        self.running = False         # Flag to indicate if the camera is running
        self.paused = False          # Flag to indicate if video feed is paused
        self.match_found = False
        self.client_image_path = None
        self.client_name = None
        self.confirmed_clients = set()  # Set to keep track of confirmed clients

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

    def update_frame(self):
        if not self.running:
            self.update_frame_id = None
            return
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Display frame
                frame_display = cv2.resize(frame, (1280, 960))
                frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_display)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                if not self.paused:
                    # Face detection logic
                    match_found, client_image_path, client_name, bbox = create_embedding(frame)
                    
                    if bbox:
                        x1,y1,x2,y2 = bbox
                        if client_name not in {name for name, _ in self.confirmed_clients}:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display frame
                    frame_display = cv2.resize(frame, (1280, 960))
                    frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_display)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

                    
                    if match_found:
                        if client_name not in {name for name, _ in self.confirmed_clients}:
                            self.match_found = True
                            self.client_image_path = client_image_path
                            self.client_name = client_name
                            self.info_label.config(text=f"Match found: {self.client_name}")
                            self.paused = True  # Set paused to True
                            self.show_client_image()  # Show the client's image before confirming
                        else:
                            print(f"Client {client_name} already confirmed, skipping.")
                    else:
                        self.info_label.config(text="Client Information: Not Detected")
            else:
                print("Failed to read frame")
        else:
            print("Capture device not opened")

        # Reschedule update_frame if the camera is still running
        if self.running:
            self.update_frame_id = self.root.after(10, self.update_frame)
        else:
            self.update_frame_id = None

    def show_client_image(self):
        if self.client_image_path:
            self.paused = True  # Pause the video feed
            client_image = cv2.imread(self.client_image_path)
            client_image = cv2.resize(client_image, (640, 480))
            client_image = cv2.cvtColor(client_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(client_image)
            imgtk = ImageTk.PhotoImage(image=img)

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
            print("Confirmation window displayed")

    def client_confirmed(self):
        print("Client confirmed")
        messagebox.showinfo("Client Login", f"{self.client_name} logged in successfully!")
        # Add the client to the confirmed clients set with a timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.confirmed_clients.add((self.client_name, timestamp))
        # Update the confirmed clients listbox
        self.update_confirmed_listbox()
        self.close_client_window()

    def client_cancelled(self):
        print("Client cancelled")
        self.info_label.config(text="Client Information: Not Detected")
        self.close_client_window()

    def close_client_window(self):
        self.top_image_window.destroy()
        self.match_found = False
        self.paused = False  # Resume video feed
        print("Resuming video feed")

    def update_confirmed_listbox(self):
        # Clear the listbox
        self.confirmed_listbox.delete(0, tk.END)
        # Insert all confirmed clients into the listbox with timestamp
        for client_name, timestamp in sorted(self.confirmed_clients):
            display_text = f"{client_name} - Confirmed at {timestamp}"
            self.confirmed_listbox.insert(tk.END, display_text)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

    # Close the database connection
    conn.close()
