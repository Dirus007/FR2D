import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import utils.deletion
import utils.viewing
import utils.loader
import utils.frame_handling
import utils.registration
import utils.redirect
import json
from testing.speed.FPS import FPSMeter


def load_settings(config_file_dir):
    try:
        with open(config_file_dir, 'r') as settings_file:
            settings_data = json.load(settings_file)

    except FileNotFoundError:
        default_settings = {
            "EAR_THRESH": 0.24,
            "BLINK_CONSEC_FRAMES": 2,
            "BLINKS_REQUIRED": 4,
            "FRAME_WIDTH": 500,
            "FACE_DISTANCE_THRESHOLD": 0.45,
            "ADMIN_PASSWORD": "admin",
            "UNMATCH_BOX_COLOR": (0, 0, 255),
            "MATCH_BOX_COLOR": (0, 255, 0),
            "NEAREST_FACE_METHOD": 'euclidean',
            "zoom_level": 100,
        }
        with open(config_file_dir, 'w') as new_settings_file:
            json.dump(default_settings, new_settings_file, indent=2)
        settings_data = default_settings
    return settings_data

script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_dir = os.path.join(script_dir, 'FR_2DConfigs.json')

settings = load_settings(config_file_dir)

COUNTER = 0
EAR_THRESH = settings.get("EAR_THRESH")
BLINK_CONSEC_FRAMES = settings.get("EAR_THRESH")
BLINKS_REQUIRED = settings.get("BLINKS_REQUIRED")
TOTAL = BLINKS_REQUIRED
FRAME_WIDTH = settings.get("FRAME_WIDTH")
proceed_button = None
FACE_DISTANCE_THRESHOLD = settings.get("FACE_DISTANCE_THRESHOLD")
ADMIN_PASSWORD = settings.get("ADMIN_PASSWORD")
register_button = None
encodings_file = "encodings.pickle"
UNMATCH_BOX_COLOR = settings.get("UNMATCH_BOX_COLOR")
MATCH_BOX_COLOR = settings.get("MATCH_BOX_COLOR")
NEAREST_FACE_METHOD = settings.get("NEAREST_FACE_METHOD")
button_config = {
        "bg": "LightGray",
        "fg": "Red",
        "font": ("Arial", 8),
    }

zoom_level = int(settings.get("zoom_level"))
previous_sno = -1

data, _ = utils.loader.load_data(encodings_file, 'normal', '')


def update_settings(config_file_dir):
    with open(config_file_dir, 'r') as settings_file:
        settings_data = json.load(settings_file)

    settings_data["EAR_THRESH"] = EAR_THRESH
    settings_data["BLINK_CONSEC_FRAMES"] = BLINK_CONSEC_FRAMES
    settings_data["COUNTER"] = COUNTER
    settings_data["BLINKS_REQUIRED"] = BLINKS_REQUIRED
    settings_data["FRAME_WIDTH"] = FRAME_WIDTH
    settings_data["FACE_DISTANCE_THRESHOLD"] = FACE_DISTANCE_THRESHOLD
    settings_data["ADMIN_PASSWORD"] = ADMIN_PASSWORD
    settings_data["UNMATCH_BOX_COLOR"] = UNMATCH_BOX_COLOR
    settings_data["NEAREST_FACE_METHOD"] = NEAREST_FACE_METHOD
    settings_data["zoom_level"] = zoom_level

    with open(config_file_dir, 'w') as new_settings_file:
        json.dump(settings_data, new_settings_file, indent=2)


def create_proceed_button():
    utils.redirect.redirect(proceed_button, TOTAL)


def create_register_button():
    global register_button
    if register_button is None:
        register_button = tk.Button(frame_controls, text="Register Face", command=register_face,
                                    bg=button_config["bg"], fg=button_config["fg"], font=button_config["font"])
        register_button.pack()


def hide_register_button():
    global register_button
    if register_button is not None:
        register_button.pack_forget()
        register_button = None


def view_people_in_db():
    global data
    utils.viewing.view_people(ADMIN_PASSWORD, window, data)


def delete_people_in_db():
    global data, encodings_file
    utils.deletion.delete_people(data, encodings_file, ADMIN_PASSWORD, window)


def update_message_show_right(sno, name):
    label_blink_count_secret.config(text=TOTAL)
    if sno == -1:
        label_message.config(text=f"Unknown Face")
    else:
        label_message.config(text=f"Welcome {name}, face identified")


def display_in_tkinter(frame):
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)


def draw_rectangle_and_name(frame, left, top, right, bottom, name, box_color):
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)


def update_counter_and_total(ear):
    global COUNTER, TOTAL
    if ear < EAR_THRESH:
        COUNTER += 1
    else:
        if COUNTER >= BLINK_CONSEC_FRAMES:
            if TOTAL > 0:
                TOTAL -= 1
        COUNTER = 0


def register_button_visibility(sno, register_button):
    if sno == -1:
        if register_button is None:
            create_register_button()
    else:
        hide_register_button()


fps_meter = FPSMeter()


def update_image():
    global COUNTER, TOTAL, proceed_button, register_button, previous_sno
    ret, frame = cap.read()
    fps_meter.record_frame_time()
    if ret:
        frame = utils.frame_handling.zoom_frame(frame, zoom_level)
        face_locations, rgb_frame, frame = utils.frame_handling.get_all_faces(frame, FRAME_WIDTH, enable_adaptive_threshold = False)
        largest_face_location = utils.frame_handling.find_most_prominent_face(face_locations)

        name = "Unknown"
        sno = -1
        ear = 0

        if largest_face_location:
            face_landmarks_list, face_encodings = utils.frame_handling.landmarks_list_and_encodings(rgb_frame, largest_face_location)
            ear = utils.frame_handling.process_eyes(frame, face_landmarks_list[0])
            update_counter_and_total(ear)

            top, right, bottom, left = largest_face_location

            encoding = face_encodings[0]

            if True:
                sno, name = utils.frame_handling.find_nearest_face(data, encoding, FACE_DISTANCE_THRESHOLD, NEAREST_FACE_METHOD)

                if sno != previous_sno:
                    TOTAL = BLINKS_REQUIRED
                    previous_sno = sno

                # Face Bounding Box
                box_color = UNMATCH_BOX_COLOR
                if TOTAL == 0 and sno != -1:
                    box_color = MATCH_BOX_COLOR
                    if proceed_button is None:
                        create_proceed_button()
                    hide_register_button()
                register_button_visibility(sno, register_button)
                draw_rectangle_and_name(frame, left, top, right, bottom, name, box_color)

        # Blink Count and EAR Text
        # blink_text = f"Blinks: {TOTAL}"
        # ear_text = f"EAR: {ear:.2f}"

        # cv2.putText(frame, blink_text, (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
        # cv2.putText(frame, ear_text, (frame.shape[1] - 120, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255, 255, 255), 1)

        # label_blink_count.config(text=blink_text)
        # label_ear.config(text=ear_text)

        update_message_show_right(sno, name)
        display_in_tkinter(frame)

    window.after(10, update_image)


def register_face():
    global data, TOTAL
    utils.registration.register_face(data, TOTAL, ADMIN_PASSWORD, cap,
                                     encodings_file, BLINKS_REQUIRED, window)

# Tkinter Window
window = tk.Tk()
window.title("Face Recognition System")
#window.configure(bg='light grey')

frame_video = tk.Frame(window)
frame_controls = tk.Frame(window)
frame_video.pack(side="left", padx=10, pady=10)
frame_controls.pack(side="right", padx=10, pady=10)

label_video = tk.Label(frame_video)
label_video.pack()

label_blink_count_secret = tk.Label(window, text="0")
label_blink_count_secret.place(relx=0.97, rely=0.97, anchor="se")

# label_blink_count = tk.Label(window, text="Blinks: 0")
# label_blink_count.place(relx=0.95, rely=0.95, anchor="se")

# label_ear = tk.Label(window, text="EAR: 0.00")
# label_ear.place(relx=0.95, rely=0.90, anchor="se")

label_message = tk.Label(frame_controls, text="", width=30, height=2)
label_message.pack()


def create_specified_buttons(frame, allowed_buttons):
    button_actions = {
        "register": ("Register Face", register_face),
        "delete": ("Delete Person", delete_people_in_db),
        "view": ("View Registered People", view_people_in_db)
    }

    for name in allowed_buttons:
        if name in button_actions:
            text, command = button_actions[name]
            button = tk.Button(frame, text=text, command=command,
                               bg=button_config["bg"], fg=button_config["fg"], font=button_config["font"])
            button.pack()


def open_settings_window():
    settings_window = tk.Toplevel(window)
    settings_window.title("Settings")
    tk.Label(settings_window, text="Zoom Level:").pack()
    zoom_level_entry = tk.Entry(settings_window)

    tk.Label(settings_window, text="Zoom Level:").pack()
    zoom_level_entry = tk.Entry(settings_window)

    zoom_level_entry.insert(0, str(zoom_level))
    zoom_level_entry.pack()

    try:
        available_cameras = utils.frame_handling.get_available_cameras()
    except Exception as e:
        tk.Label(settings_window, text=f"Error: {str(e)}").pack()
        available_cameras = []

    if available_cameras:
        tk.Label(settings_window, text="Select Camera:").pack()
        camera_var = tk.StringVar()
        camera_var.set(str(available_cameras[0]))
        camera_dropdown = tk.OptionMenu(settings_window, camera_var, *map(str, available_cameras))
        camera_dropdown.pack()

    def apply_settings():
        global zoom_level, cap
        zoom_level = int(zoom_level_entry.get())
        if cap:
            cap.release()

        if available_cameras:
            selected_camera = int(camera_var.get())
            cap = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)

        settings_window.destroy()

    tk.Button(settings_window, text="Apply", command=apply_settings).pack()


gear_img_dir = os.path.join(script_dir, "images\gear.jpg")
original_icon = Image.open(gear_img_dir)
resized_icon = original_icon.resize((30, 30))
gear_icon = ImageTk.PhotoImage(resized_icon)
gear_button = tk.Button(window, image=gear_icon, command=open_settings_window)
gear_button.place(relx=0.97, rely=0.03, anchor="ne")


button_list = []
create_specified_buttons(frame_controls, button_list)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
update_image()
window.mainloop()


cap.release()
cv2.destroyAllWindows()

update_settings(config_file_dir)
fps_meter.plot_graph()
