# 2D Face Recognition System

## Abstract
Our 2D Face Recognition System leverages the Python ecosystem to deliver a robust solution for facial recognition needs. It incorporates Tkinter for the graphical user interface, OpenCV for image processing, and the `face_recognition` library for advanced facial recognition tasks. With a focus on security, this system employs anti-spoofing techniques through blink detection, ensuring that authentication is both secure and reliable.

## Key Features
- **Precise Detection**: The system ensures accurate detection and identification of individuals.
- **Anti-Spoofing**: Implements blink detection to differentiate between real persons and static images.
- **User Management**: Facilitates new face registrations and handles data management effectively.
- **Modular Design**: Adopts a modular architecture for easy maintainability and future enhancements.

## Installation
### Clone the repository
```bash
git clone https://github.com/Dirus007/FR2D.git
```
### Navigate to the project directory
```bash
cd FR2D
```
### Install the required dependencies
```bash
pip install -r requirements.txt
```

## Things that can be implemented in the future
- **Using ANNOY for Faster Searches**: Implementing Approximate Nearest Neighbour Oh Yeah (ANNOY) can significantly speed up the process of searching and matching faces in the database. This is particularly beneficial when the database size grows.
- **User-Friendly Interface Improvements**: Enhance the GUI for a more intuitive and user-friendly experience.
- **Customizable Settings**: Allow users to customize settings like blink threshold, camera selection, and alert preferences.
- **Enhanced Data Privacy Measures**: Implement robust data encryption and privacy measures to protect user data.
- **Continuous Learning and Updates**: Implement machine learning algorithms that continuously learn and improve accuracy over time.
