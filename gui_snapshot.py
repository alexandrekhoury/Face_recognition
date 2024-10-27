import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
import sqlite3
import time
import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

# Initialize MTCNN and FaceNet model
mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define minimum face size and directories for embeddings and images
MIN_FACE_SIZE = (80, 80)
embedding_dir = "client_data/embeddings"
image_dir = "client_data/images"
os.makedirs(embedding_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Connect to the database
conn = sqlite3.connect('clients.db')
cursor = conn.cursor()

class FaceCaptureApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        # Setup the GUI layout
        self.setWindowTitle("Face Capture Confirmation")
        self.setGeometry(100, 100, 400, 400)

        self.client_name = ""
        self.embedding_saved = False
        self.face_region = None
        self.frame = None
        
        # Label for the client name
        self.name_label = QtWidgets.QLabel("Client Name: ", self)
        self.name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.name_label.setFont(QtGui.QFont("Arial", 16))
        
        # Label for displaying the captured image
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # Yes and No buttons for confirmation
        self.yes_button = QtWidgets.QPushButton("Yes", self)
        self.no_button = QtWidgets.QPushButton("No", self)
        
        # Connect buttons to their actions
        self.yes_button.clicked.connect(self.save_client_info)
        self.no_button.clicked.connect(self.retake_snapshot)
        
        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.name_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.yes_button)
        layout.addWidget(self.no_button)
        self.setLayout(layout)
        
        self.prompt_for_name()
    
    def prompt_for_name(self):
        """Prompt the user for the client's name."""
        text, ok = QtWidgets.QInputDialog.getText(self, "Input", "Enter the client's name:")
        if ok and text:
            self.client_name = text.replace(" ", "_")
            self.start_capture()
        else:
            self.close()
    
    def start_capture(self):
        """Start the video capture to detect and display a face."""
        cap = cv2.VideoCapture(4)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Could not open video stream.")
            self.close()
            return
        
        while not self.embedding_saved:
            ret, frame = cap.read()
            if not ret:
                QtWidgets.QMessageBox.critical(self, "Error", "Failed to capture image.")
                break
            
            # Detect face
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    width, height = x2 - x1, y2 - y1

                    if width >= MIN_FACE_SIZE[0] and height >= MIN_FACE_SIZE[1]:
                        self.face_region = frame[y1:y2, x1:x2]
                        self.frame = frame
                        self.embedding_saved = True
                        self.display_captured_image()
                        break
            
            cv2.imshow('Live Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def display_captured_image(self):
        """Display the captured face and client name in the GUI."""
        # Update the client name label
        self.name_label.setText(f"Client Name: {self.client_name}")
        
        # Convert captured face image to QPixmap and display in QLabel
        face_image = cv2.cvtColor(self.face_region, cv2.COLOR_BGR2RGB)
        h, w, ch = face_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(face_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def save_client_info(self):
        """Save the face, embedding, and client info to the database."""
        face_resized = cv2.resize(self.face_region, (160, 160))
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
        face_tensor = face_tensor.to(torch.float32)

        # Generate embedding
        with torch.no_grad():
            embedding = model(face_tensor).detach().numpy()

        # Create unique filenames
        timestamp = int(time.time())
        image_filename = os.path.join(image_dir, f"{self.client_name}_{timestamp}.jpg")
        embedding_filename = os.path.join(embedding_dir, f"{self.client_name}_{timestamp}.npy")

        # Save image and embedding
        cv2.imwrite(image_filename, self.frame)
        np.save(embedding_filename, embedding)

        # Insert client info into the database
        cursor.execute('''
            INSERT INTO clients (name, image_path, embedding_path)
            VALUES (?, ?, ?)
        ''', (self.client_name, image_filename, embedding_filename))
        conn.commit()

        QtWidgets.QMessageBox.information(self, "Info", f"Client '{self.client_name}' has been saved successfully!")
        self.close()

    def retake_snapshot(self):
        """Retake the snapshot by restarting the capture process."""
        self.embedding_saved = False
        self.start_capture()

# Run the PyQt application
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    face_capture_app = FaceCaptureApp()
    face_capture_app.show()
    sys.exit(app.exec_())
    
# Close the database connection when done
conn.close()
