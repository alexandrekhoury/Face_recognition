# Face Recognition System

This project is a face recognition system that allows you to:

- Register new clients by capturing their facial data.
- Detect and recognize clients in real-time using a live camera feed.
- Automatically confirm client identities with the option to toggle between automatic and manual confirmation modes.

The system uses machine learning models for face detection and recognition, providing an interactive GUI for ease of use.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Initializing the System](#initializing-the-system)
- [Registering New Clients](#registering-new-clients)
- [Detecting and Recognizing Clients](#detecting-and-recognizing-clients)
- [Usage Notes](#usage-notes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Client Registration**: Capture facial data and register new clients into the system using a simple GUI.
- **Real-time Detection**: Use the live camera feed to detect and recognize clients in real-time.
- **Automatic Confirmation**: Optionally enable automatic confirmation after a timeout period.
- **Manual Override**: Manually confirm or cancel client recognition.
- **Client List Display**: View a list of all recognized clients during a session.
- **Data Persistence**: Store client data, including embeddings and images, in a structured directory and database.

## Prerequisites

- **Operating System**: Linux, macOS, or Windows.
- **Python**: Version 3.6 or higher.
- **Camera**: A webcam or camera device connected to your system.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages using pip.

```bash
pip install -r requirements.txt
```

**Note**: Ensure that you have the required system packages for OpenCV and PyTorch. For example, on Ubuntu:

```bash
sudo apt-get install libgl1-mesa-glx
```

## Project Structure

```bash
.
├── client_data
│   ├── embeddings      # Directory for storing client embeddings
│   └── images          # Directory for storing client images
├── clients.db          # SQLite database for client information
├── create_db.py        # Script to initialize the database
├── gui_detect_automatic.py  # Main application script for detection
├── gui_take_snapshot.py     # Script for registering new clients
├── requirements.txt    # List of Python dependencies
├── README.md           # Project documentation
└── venv                # Virtual environment directory
```

## Initializing the System

Before using the system, you need to initialize the database and ensure the required directories are set up.

### 1. Initialize the Database

Run the `create_db.py` script to create the `clients.db` SQLite database.

```bash
python create_db.py
```

This script will create a `clients.db` file with a `clients` table to store client information.

### 2. Create Necessary Directories

Ensure that the following directories exist:

- `client_data/embeddings`
- `client_data/images`

If they do not exist, create them manually or ensure that the scripts can create them automatically.

## Registering New Clients

To register new clients, use the `gui_take_snapshot.py` script.

### Steps:

#### 1. Run the Registration Script

```bash
python gui_take_snapshot.py
```

#### 2. Enter Client's Name

- A dialog will appear asking for the client's name.
- Enter the full name of the client and press OK.

#### 3. Capture the Client's Face

- A window will display the live camera feed with the client's name at the top.
- Ensure the client's face is clearly visible in the frame.
- The system will detect the face and draw a rectangle around it.

#### 4. Confirm the Image

- A confirmation dialog will appear asking if you're satisfied with the captured image.
  - **Yes**: The system will save the client's image and embedding, and add the client to the database.
  - **No**: The system will resume the camera feed to retake the image.

#### 5. Success Message

- If confirmed, a success message will appear indicating the client was added successfully.

#### 6. Exit the Application

- After registration, the application will close automatically.

**Note**: Ensure proper lighting and positioning for optimal face detection.

## Detecting and Recognizing Clients

To detect and recognize registered clients, use the `gui_detect_automatic.py` script.

### Steps:

#### 1. Run the Detection Script

```bash
python gui_detect_automatic.py
```

#### 2. Start the Camera

- Click on the **Start Camera** button to begin the live feed.
- The camera feed will appear on the left side of the window.

#### 3. Monitor Client Recognition

- The system will detect faces in real-time.
- When a registered client is detected:
  - A rectangle will appear around their face.
  - The client's name will be displayed.
  - If **Auto-Confirm** is enabled, the system will automatically confirm the client after 3 seconds unless canceled.
  - If **Auto-Confirm** is disabled, a confirmation window will appear for manual confirmation.

#### 4. View Confirmed Clients

- The right side of the window displays a list of confirmed clients with timestamps.
- This list updates as new clients are confirmed.

#### 5. Toggle Auto-Confirm

- Use the **Auto-Confirm ON/OFF** button to enable or disable automatic confirmation.
  - **Auto-Confirm ON**: Clients are automatically confirmed after a timeout.
  - **Auto-Confirm OFF**: Manual confirmation is required.

#### 6. Stop the Camera

- Click on the **Stop Camera** button to halt the live feed.

#### 7. Exit the Application

- Click on the **Quit** button to close the application.

## Usage Notes

- **Multiple Clients**: The system can recognize multiple clients in the same session.
- **Session Persistence**: The list of confirmed clients resets when the application restarts.
- **Data Storage**: Client images and embeddings are stored in the `client_data` directory, and client information is stored in `clients.db`.
- **Face Recognition Model**: The system uses the InceptionResnetV1 model pretrained on the VGGFace2 dataset.

## Troubleshooting

- **Camera Not Opening**: Ensure that your camera device is connected and not being used by another application.
- **Face Not Detected**: Check lighting conditions and make sure the face is within the camera frame.
- **Dependencies Issues**: Verify that all dependencies are installed correctly, especially PyTorch and OpenCV.
- **Database Errors**: Ensure the `clients.db` file exists and the database schema is initialized.

## License

This project is licensed under the MIT License.

**Disclaimer**: This system is intended for educational purposes. Ensure compliance with privacy laws and regulations when collecting and processing biometric data.
