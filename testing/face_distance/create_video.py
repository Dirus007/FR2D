import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import os

is_recording = False
video_frames = []
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 500
FRAME_HEIGHT = 300
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "recording.avi")


def capture_video():
    global is_recording, video_frames, cap
    while is_recording:
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_WIDTH))
            video_frames.append(resized_frame)


def start_recording():
    global is_recording
    is_recording = True
    threading.Thread(target=capture_video, daemon=True).start()
    update_video_display()


def stop_recording():
    global is_recording
    is_recording = False


def save_recording():
    global video_frames
    if video_frames:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (FRAME_WIDTH, FRAME_WIDTH))
        for frame in video_frames:
            out.write(frame)
        out.release()
        video_frames = []


def update_video_display():
    global label_video
    if is_recording:
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_WIDTH))
            cv2image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            label_video.imgtk = imgtk
            label_video.configure(image=imgtk)
        window.after(10, update_video_display)


def on_closing():
    global cap
    if is_recording:
        stop_recording()
    cap.release()
    window.destroy()


# Tkinter Window Setup
window = tk.Tk()
window.title("Video Recorder")

frame_video = tk.Frame(window)
frame_video.pack(side="left", padx=10, pady=10)

label_video = tk.Label(frame_video)
label_video.pack()

frame_controls = tk.Frame(window)
frame_controls.pack(side="right", padx=10, pady=10)

start_button = tk.Button(frame_controls, text="Start", command=start_recording)
start_button.pack()

stop_button = tk.Button(frame_controls, text="Stop", command=stop_recording)
stop_button.pack()

save_button = tk.Button(frame_controls, text="Save", command=save_recording)
save_button.pack()

window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
