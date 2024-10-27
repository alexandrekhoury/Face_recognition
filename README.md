
# Face Recognition and Snapshot System

This project is a Python-based face recognition system designed to capture snapshots of clients, confirm their identity, and store their facial embeddings in a database. The project uses `PyQt` or `tkinter` for GUI, `OpenCV` for image processing, and `facenet_pytorch` for face recognition with the FaceNet model. It’s ideal for real-time applications like automated gym check-ins.

## Directory Structure

- **`client_data/`**: Contains the subdirectories for saved client images and embeddings.
  - **`embeddings/`**: Stores numpy files (`.npy`) containing the facial embeddings of each client.
  - **`images/`**: Stores images (`.jpg`) of each client captured by the system.

- **`clients.db`**: SQLite database file containing client information, including names, image paths, and embedding paths.

- **`face_embeddings/`**: (If applicable) Contains additional embeddings if needed.

- **`create_db.py`**: Script for setting up the `clients.db` SQLite database, creating the required tables to store client information.

- **`gui_snapshot.py`**: Main GUI application that captures client snapshots using `PyQt`. Displays the captured image and asks for user confirmation to save or retake the image. Shows the client’s name and saves the image and embedding if confirmed.

- **`live_camera.py`**: Script to detect and recognize clients in real-time using the camera feed. It compares each captured face with stored embeddings and notifies if a match is found.

- **`take_snapshot.py`**: Non-GUI script to capture a client’s face, confirm the image with a simple command-line prompt, and save the embedding and image. Useful for environments without GUI support.

- **`README.md`**: This README file that provides an overview and instructions for the project.

## Setup

### 1. Install Dependencies

Ensure you have Python 3.7+ installed. Use the following commands to install the necessary libraries:

```bash
pip install opencv-python-headless facenet-pytorch torch numpy sqlite3 pillow PyQt5
```

### 2. Set Up the Database

Run `create_db.py` to set up the `clients.db` SQLite database with the necessary tables:

```bash
python create_db.py
```

### 3. Run the GUI Application

To capture a new client snapshot with a GUI that asks for confirmation before saving, run:

```bash
python gui_snapshot.py
```

Follow the prompts to enter the client’s name, capture the snapshot, and confirm if you want to save it.

### 4. Run Real-Time Face Recognition

To detect and recognize faces in real-time, use `live_camera.py`. This script opens a live camera feed and compares detected faces with stored embeddings.

```bash
python live_camera.py
```

### 5. Command-Line Snapshot Capture

If you prefer a command-line interface to capture snapshots, use `take_snapshot.py`:

```bash
python take_snapshot.py
```

This script will prompt for the client’s name and ask for confirmation after each capture.

## Usage Overview

1. **Capture New Clients**: Use either `gui_snapshot.py` for a GUI-based experience or `take_snapshot.py` for a command-line experience. Both scripts capture and save the client’s image and embedding in `client_data/`.

2. **Real-Time Recognition**: Use `live_camera.py` to check clients in real-time. The script will detect and match clients with the saved embeddings in `client_data/embeddings`.

3. **Database**: All client data, including names, image paths, and embedding paths, are stored in `clients.db`. This allows for easy retrieval and matching.

## Troubleshooting

- **Qt Warnings**: If you encounter `QObject::moveToThread` warnings, try setting environment variables to disable specific plugins or use the `opencv-python-headless` package to avoid Qt GUI elements.

- **Cannot Find Qt Plugins**: Set `QT_PLUGIN_PATH` or install `libxcb-xinerama0` if you encounter `xcb` plugin issues on Linux.

- **GUI Issues**: If the GUI fails to start or you receive "Could not load Qt platform plugin," consider reinstalling `PyQt5` or using `tkinter` as a simpler GUI option.

## Future Improvements

- **Add Client Management**: Add features to update or delete client data from the database.
- **Advanced Matching**: Improve face matching algorithms or adjust thresholds for better accuracy in different lighting conditions.
- **Enhanced GUI**: Add more controls and customization to the GUI for a better user experience.

