from tkinter import simpledialog, messagebox
import cv2
import imutils
import pickle
import face_recognition


def register_face(data, TOTAL, ADMIN_PASSWORD, cap, encodings_file, BLINKS_REQUIRED, window):
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
