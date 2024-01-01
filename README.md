# Simple 2D Face Recognition System

## Description
This facial recognition system is developed in Python, using Tkinter for the graphical user interface, OpenCV for image processing, and the `face_recognition` library for facial recognition tasks. It provides functionalities to register, view, and delete faces from a database. Additionally, it uses blink detection for user authentication.

## Features
- **Face Registration**: Allows users to register a new face in the system.
- **Face Deletion**: Enables the removal of a registered face from the system.
- **View Registered Faces**: Displays all faces stored in the database.
- **Authentication**: Users are authenticated based on face recognition and blink detection.

## Requirements
- Python 3.8
- OpenCV (`cv2`)
- face_recognition
- NumPy
- Pillow
- Tkinter
- imutils

## Installation
Install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

## Things that can be implemented in the future
- **Using ANNOY for Faster Searches**: Implementing Approximate Nearest Neighbour Oh Yeah (ANNOY) can significantly speed up the process of searching and matching faces in the database. This is particularly beneficial when the database size grows.
- **User-Friendly Interface Improvements**: Enhance the GUI for a more intuitive and user-friendly experience.
- **Customizable Settings**: Allow users to customize settings like blink threshold, camera selection, and alert preferences.
- **Enhanced Data Privacy Measures**: Implement robust data encryption and privacy measures to protect user data.
- **Continuous Learning and Updates**: Implement machine learning algorithms that continuously learn and improve accuracy over time.
