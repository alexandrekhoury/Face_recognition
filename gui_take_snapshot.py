import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
import sqlite3
import time
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Initialize MTCNN and FaceNet model
mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define minimum face size and directories
MIN_FACE_SIZE = (80, 80)
embedding_dir = "client_data/embeddings"
image_dir = "client_data/images"
os.makedirs(embedding_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Connect to the database
conn = sqlite3.connect('clients.db')
cursor = conn.cursor()

def capture_and_save_client_info(name, frame, face_region):
    """Capture the face, save embedding and image, and insert info into the database."""
    face_resized = cv2.resize(face_region, (160, 160))
    face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
    face_tensor = face_tensor.to(torch.float32)

    with torch.no_grad():
        embedding = model(face_tensor).detach().numpy()
    
    timestamp = int(time.time())
    name = name.replace(" ","_")
    image_filename = os.path.join(image_dir, f"{name}_{timestamp}.jpg")
    embedding_filename = os.path.join(embedding_dir, f"{name}_{timestamp}.npy")

    cv2.imwrite(image_filename, frame)
    np.save(embedding_filename, embedding)

    cursor.execute('''
        INSERT INTO clients (name, image_path, embedding_path)
        VALUES (?, ?, ?)
    ''', (name, image_filename, embedding_filename))
    conn.commit()
    
    messagebox.showinfo("Success", f"Client '{name}' added successfully!")
    return True

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.capture = cv2.VideoCapture(4)
        self.video_label = tk.Label(window)
        self.video_label.pack()

        self.client_name = simpledialog.askstring("Input", "Enter the client's name:")
        self.embedding_saved = False
        self.camera_paused = False  # Flag to control when the camera feed is paused
        self.last_capture_time = time.time()

        # Resize the window to double the usual size
        self.window.geometry("1280x960")  # Set window size explicitly

        self.update_frame()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update_frame(self):
        if not self.camera_paused:  # Only update if the camera is not paused
            ret, frame = self.capture.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image.")
                self.on_closing()
           
            # Double the frame size
            frame = cv2.resize(frame, (1280, 960))

            # Draw the client name at the top of the frame
            cv2.putText(frame, self.client_name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display the live video frame in the Tkinter label
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.video_label.configure(image=tk_image)
            self.video_label.image = tk_image

            # Only check for faces once per second
            if time.time() - self.last_capture_time >= 1:
                self.detect_and_confirm_face(frame)
                self.last_capture_time = time.time()

        if not self.embedding_saved:
            self.window.after(30, self.update_frame)

    def detect_and_confirm_face(self, frame):
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                width, height = x2 - x1, y2 - y1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Convert frame to RGB format for Tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                tk_image = ImageTk.PhotoImage(pil_image)

                self.video_label.configure(image=tk_image)
                self.video_label.image = tk_image



                # Only process if the face is large enough
                if width >= MIN_FACE_SIZE[0] and height >= MIN_FACE_SIZE[1]:
                    face_region = frame[y1:y2, x1:x2]

                    # Pause the camera, show confirmation dialog, and resume after response
                    self.camera_paused = True
                    self.show_custom_confirmation_dialog(frame, face_region)

                    if self.embedding_saved:
                        self.on_closing()
                    break
    def show_custom_confirmation_dialog(self, frame, face_region):
        """Display a custom confirmation dialog positioned relative to the main window."""
        dialog = tk.Toplevel(self.window)
        dialog.title("Confirm Image")

        # Get main window position and set dialog position relative to it
        main_x = self.window.winfo_x()
        main_y = self.window.winfo_y()
        dialog.geometry(f"+{main_x + 200}+{main_y + 100}")  # Offset to the right and slightly down from the main window

        # Make the dialog modal
        dialog.transient(self.window)  # Make the dialog appear in front of the main window
        dialog.grab_set()  # Capture all events for this dialog

        label = tk.Label(dialog, text="Are you satisfied with this image?")
        label.pack(pady=10)

        # Confirmation buttons
        yes_button = tk.Button(dialog, text="Yes", command=lambda: self.on_confirm(dialog, frame, face_region, True))
        no_button = tk.Button(dialog, text="No", command=lambda: self.on_confirm(dialog, frame, face_region, False))
        yes_button.pack(side=tk.LEFT, padx=10, pady=5)
        no_button.pack(side=tk.RIGHT, padx=10, pady=5)

        # Wait for the dialog to close
        self.window.wait_window(dialog)
    def on_confirm(self, dialog, frame, face_region, confirmed):
        """Handle the result of the confirmation dialog."""
        dialog.destroy()  # Close the dialog window

        if confirmed:
            self.embedding_saved = capture_and_save_client_info(self.client_name, frame, face_region)
        else:
            messagebox.showinfo("Info", "Retaking image...")

        self.camera_paused = False  # Resume camera feed after dialog
    def on_closing(self):
        self.capture.release()
        self.window.destroy()
        conn.close()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Face Detection and Capture")
