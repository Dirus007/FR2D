import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import pickle
import imutils
import os
import subprocess
import sys

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


EAR_THRESH = 0.20
BLINK_CONSEC_FRAMES = 1
COUNTER = 0
BLINKS_REQUIRED = 3
TOTAL = BLINKS_REQUIRED
proceed_button = None
FACE_DISTANCE_THRESHOLD = 0.45
ADMIN_PASSWORD = "admin"
register_button = None


def create_proceed_button():
    global proceed_button, TOTAL
    if proceed_button is None:
        if TOTAL == 0:
            messagebox.showinfo("Authentication Successful", "You are authenticated !")
            # path = r"C:\Users\Mukul  Dev\OneDrive\Desktop\03 John Wick Chapter 3 Parabellum - Action 2019 Eng Subs 720p [H264-mp4].mp4"
            # run_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
            path = r"C:\Users\Mukul  Dev\OneDrive\Desktop\Markdown to Table\run.pyw"
            run_path = r"C:\Users\Mukul  Dev\AppData\Local\Programs\Python\Python311\pythonw.exe"
            subprocess.Popen([run_path, path])
            sys.exit()


            # proceed_button = tk.Button(frame_controls, text="Proceed", command=show_message_good)
            # proceed_button.pack()
        # else:
        #     proceed_button = tk.Button(frame_controls, text="Proceed", command=show_message_bad)
        #     proceed_button.pack()


def create_register_button():
    global register_button
    if register_button is None:
        register_button = tk.Button(frame_controls, text="Register Face", command=register_face)
        register_button.pack()


def hide_register_button():
    global register_button
    if register_button is not None:
        register_button.pack_forget()
        register_button = None


# def show_message_good():
#     messagebox.showinfo("Authentication Successful", "You are authenticated !")
#
#
# def show_message_bad():
#     messagebox.showinfo("Authentication Not Successful", "You are not authenticated !")


def view_people_in_db():
    admin_password = simpledialog.askstring("Password", "Enter admin password:", parent=window, show="*")
    if admin_password == ADMIN_PASSWORD:
        view_window = tk.Toplevel(window)
        view_window.title("Registered People")

        text_area = tk.Text(view_window, height=15, width=50)
        text_area.pack(padx=10, pady=10)
        scroll = tk.Scrollbar(view_window, command=text_area.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.configure(yscrollcommand=scroll.set)

        if data["service_number"]:
            full_list = ""
            for service_number, name in zip(data["service_number"], data["names"]):
                full_list += f"Service Number: {service_number}, Name: {name}\n"
            text_area.insert(tk.END, f"People in the Database:\n{full_list}")
        else:
            text_area.insert(tk.END, "No people registered in the database.")

        text_area.config(state=tk.DISABLED)
    elif admin_password:
        messagebox.showerror("Error", "Incorrect admin password.")


def delete_people_in_db():
    admin_password = simpledialog.askstring("Password", "Enter admin password:", parent=window, show="*")
    if admin_password == ADMIN_PASSWORD:
        service_number_to_delete = simpledialog.askstring("Delete Person", "Enter the service number of the person to delete:", parent=window)
        # name_to_delete = simpledialog.askstring("Delete Person", "Enter the name of the person to delete:", parent=window)
        if service_number_to_delete and service_number_to_delete in data["service_number"]:
            index = data["service_number"].index(service_number_to_delete)
            name_to_delete = data["names"][index]
            del data["names"][index]
            del data["encodings"][index]
            del data["service_number"][index]

            with open(encodings_file, "wb") as file:
                pickle.dump(data, file)
            messagebox.showinfo("Success", f"Removed {name_to_delete} from the database.")
        elif service_number_to_delete:
            messagebox.showwarning("Not Found", f"{service_number_to_delete} not in the database.")
    elif admin_password:
        messagebox.showerror("Error", "Incorrect admin password.")


encodings_file = "encodings.pickle"
if os.path.exists(encodings_file):
    print("Loading face encodings...")
    data = pickle.loads(open(encodings_file, "rb").read())
else:
    print("Starting with an empty dataset.")
    data = {"encodings": [], "service_number": [], "names": []}


window = tk.Tk()
window.title("Face Recognition System")


frame_video = tk.Frame(window)
frame_controls = tk.Frame(window)
frame_video.pack(side="left", padx=10, pady=10)
frame_controls.pack(side="right", padx=10, pady=10)


label_video = tk.Label(frame_video)
label_video.pack()

label_blink_count_secret = tk.Label(window, text="0")
label_blink_count_secret.place(relx=0.97, rely=0.97, anchor="se")

#label_blink_count = tk.Label(window, text="Blinks: 0")
#label_blink_count.place(relx=0.95, rely=0.95, anchor="se")

#label_ear = tk.Label(window, text="EAR: 0.00")
#label_ear.place(relx=0.95, rely=0.90, anchor="se")


label_message = tk.Label(frame_controls, text="", width=30, height=2)
label_message.pack()


def update_image():
    global COUNTER, TOTAL, proceed_button, register_button
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

            # Processing for the largest face only
            for face_landmarks in face_landmarks_list:
                for eye in ['left_eye', 'right_eye']:
                    for point in face_landmarks[eye]:
                        cv2.circle(frame, point, 1, (255, 255, 0), -1)

                leftEye = np.array(face_landmarks['left_eye'])
                rightEye = np.array(face_landmarks['right_eye'])
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= BLINK_CONSEC_FRAMES:
                        if TOTAL > 0 :
                            TOTAL -= 1
                    COUNTER = 0

            top, right, bottom, left = largest_face_location
            name = "Unknown"

            for encoding in face_encodings:
                if data["encodings"]:
                    face_distances = face_recognition.face_distance(data["encodings"], encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < FACE_DISTANCE_THRESHOLD:
                            name = data["names"][best_match_index]

                # Face Bounding Box
                # BGR
                box_color = (0, 0, 255)
                if TOTAL == 0 and name != "Unknown":
                    box_color = (0, 255, 0)
                    if proceed_button is None:
                        create_proceed_button()
                    hide_register_button()

                if name == "Unknown":
                    if register_button is None:
                        create_register_button()
                else:
                    hide_register_button()

                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)
                names.append(name)

        # Blink Count and EAR Text
        blink_text = f"Blinks: {TOTAL}"
        ear_text = f"EAR: {ear:.2f}"

        # cv2.putText(frame, blink_text, (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
        # cv2.putText(frame, ear_text, (frame.shape[1] - 120, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255, 255, 255), 1)

        # label_blink_count.config(text=blink_text)
        # label_ear.config(text=ear_text)

        label_blink_count_secret.config(text=TOTAL)

        if len(names) > 0:
            if names[0] == "Unknown":
                label_message.config(text=f"Unknown Face")
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
    global data, TOTAL
    admin_password = simpledialog.askstring("Password", "Enter admin password:", parent=window, show="*")
    if admin_password == ADMIN_PASSWORD:
        ret, frame = cap.read()
        if ret:
            small_frame = imutils.resize(frame, width=500)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]

                service_number = simpledialog.askstring("Input Serial Number", "Enter your service number:", parent=window)
                if service_number:
                    if service_number.isdigit() and len(service_number) in [5, 6]:
                        if service_number in data["names"]:
                            messagebox.showerror("Error", f"Service number {service_number} is already registered.")
                        else:
                            name = simpledialog.askstring("Input Name", "Enter your Name:", parent=window)
                            data["encodings"].append(face_encoding)
                            data["service_number"].append(service_number)
                            data["names"].append(name)

                            with open(encodings_file, "wb") as file:
                                file.write(pickle.dumps(data))

                            messagebox.showinfo("Success", f"Registered {name}'s face.")
                    else:
                        messagebox.showerror("Error", "Service number must be numeric and 5 or 6 digits long.")
                else:
                    messagebox.showinfo("Cancelled", "Registration Cancelled.")
            else:
                messagebox.showerror("Error", "No face detected. Try again.")
        else:
            messagebox.showerror("Error", "Incorrect admin password.")
        TOTAL = BLINKS_REQUIRED

# button_register = tk.Button(frame_controls, text="Register Face", command=register_face)
# button_register.pack()

# button_delete = tk.Button(frame_controls, text="Delete Person", command=delete_people_in_db)
# button_delete.pack()

# button_view = tk.Button(frame_controls, text="View Registered People", command=view_people_in_db)
# button_view.pack()

cap = cv2.VideoCapture(0)
update_image()
window.mainloop()


cap.release()
cv2.destroyAllWindows()
