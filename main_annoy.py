import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import pickle
import imutils
import os
from annoy import AnnoyIndex

f = 128  # Dimension of face encodings

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESH = 0.20
BLINK_CONSEC_FRAMES = 1
COUNTER = 0
TOTAL = 0
BLINKS_REQUIRED = 2
FACE_DISTANCE_THRESHOLD = 0.5

proceed_button = None

def create_proceed_button():
    global proceed_button
    if proceed_button is None:
        proceed_button = tk.Button(frame_controls, text="Proceed")
        proceed_button.pack()

def create_annoy_index(data):
    new_annoy_index = AnnoyIndex(f, 'euclidean')
    for i, encoding in enumerate(data["encodings"]):
        new_annoy_index.add_item(i, encoding)
    new_annoy_index.build(10)
    # Ensure directory exists or use an absolute path
    index_path = os.path.join(os.getcwd(), 'face_encodings.ann')
    new_annoy_index.save(index_path)
    return new_annoy_index

# Load or create data and annoy index
encodings_file = "encodings.pickle"
if os.path.exists(encodings_file):
    print("Loading face encodings...")
    data = pickle.loads(open(encodings_file, "rb").read())
else:
    print("Starting with an empty dataset.")
    data = {"encodings": [], "names": []}

annoy_index = create_annoy_index(data)

# Initialize window
window = tk.Tk()
window.title("Face Recognition System")

frame_video = tk.Frame(window)
frame_controls = tk.Frame(window)
frame_video.pack(side="left", padx=10, pady=10)
frame_controls.pack(side="right", padx=10, pady=10)

label_video = tk.Label(frame_video)
label_video.pack()

label_message = tk.Label(frame_controls, text="")
label_message.pack()

# Webcam capture
cap = cv2.VideoCapture(0)

def update_image():
    global COUNTER, TOTAL, proceed_button
    ret, frame = cap.read()

    if ret:
        frame = imutils.resize(frame, width=500)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        largest_face_area = 0
        largest_face_location = None

        # Identify the most prominent face only
        for (top, right, bottom, left) in face_locations:
            face_area = (bottom - top) * (right - left)
            if face_area > largest_face_area:
                largest_face_area = face_area
                largest_face_location = (top, right, bottom, left)

        names = []
        ear = 0

        if largest_face_location:
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame, [largest_face_location])
            face_encodings = face_recognition.face_encodings(rgb_frame, [largest_face_location])

            for face_landmarks in face_landmarks_list:
                for eye in ['left_eye', 'right_eye']:
                    for point in face_landmarks[eye]:
                        cv2.circle(frame, point, 1, (0, 255, 0), -1)

                leftEye = np.array(face_landmarks['left_eye'])
                rightEye = np.array(face_landmarks['right_eye'])
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= BLINK_CONSEC_FRAMES:
                        TOTAL += 1
                    COUNTER = 0

            top, right, bottom, left = largest_face_location
            name = "Unknown"

            if face_encodings:
                encoding = face_encodings[0]
                nearest_neighbors = annoy_index.get_nns_by_vector(encoding, 1, include_distances=True)
                if nearest_neighbors and len(nearest_neighbors[1]) > 0:
                    nearest_index, distance = nearest_neighbors
                    if distance[0] < FACE_DISTANCE_THRESHOLD:
                        name = data["names"][nearest_index[0]]

            # Face Bounding Box
            box_color = (0, 0, 255)
            if TOTAL >= BLINKS_REQUIRED and not name == "Unknown":
                box_color = (0, 255, 0)
                if proceed_button is None:
                    create_proceed_button()

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)
            names.append(name)

        # Blink Count and EAR Text
        blink_text = f"Blinks: {TOTAL}"
        ear_text = f"EAR: {ear:.2f}"
        cv2.putText(frame, blink_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, ear_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        if len(names) > 0:
            if names[0] == "Unknown":
                label_message.config(text="Unknown Face")
            else:
                label_message.config(text=f"Welcome {names[0]}, face identified")

        # Display in Tkinter
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)
    window.after(10, update_image)

def register_face():
    global data, annoy_index
    ret, frame = cap.read()
    if ret:
        small_frame = imutils.resize(frame, width=500)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)

        largest_face_area = 0
        largest_face_location = None

        # Identify the most prominent face only
        for (top, right, bottom, left) in face_locations:
            face_area = (bottom - top) * (right - left)
            if face_area > largest_face_area:
                largest_face_area = face_area
                largest_face_location = (top, right, bottom, left)

        if largest_face_location:
            face_encoding = face_recognition.face_encodings(rgb_small_frame, [largest_face_location])[0]

            name = simpledialog.askstring("Input", "What's your name?", parent=window)
            if name:
                if name in data["names"]:
                    label_message.config(text="Name already exists. Please use a different name.")
                else:
                    data["encodings"].append(face_encoding)
                    data["names"].append(name)

                    # annoy_index = create_annoy_index(data)
                    with open(encodings_file, "wb") as file:
                        file.write(pickle.dumps(data))

                    label_message.config(text=f"Registered {name}'s face.")
            else:
                label_message.config(text="Registration Cancelled.")
        else:
            label_message.config(text="No face detected. Try again.")

button_register = tk.Button(frame_controls, text="Register Face", command=register_face)
button_register.pack()

update_image()
window.mainloop()

cap.release()
cv2.destroyAllWindows()


def view_people_in_db():
    for people in data["names"]:
        print(people)


def delete_people_in_db(name):
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as file:
            prev_data = pickle.load(file)

        if name in prev_data["names"]:
            index = prev_data["names"].index(name)
            del prev_data["names"][index]
            del prev_data["encodings"][index]

            with open(encodings_file, "wb") as file:
                pickle.dump(prev_data, file)
            print(f"Removed {name} from the database.")

        else:
            print(f"{name} not in the database.")


delete_people_in_db("New")
view_people_in_db()