import subprocess
from tkinter import messagebox
import sys


def redirect(proceed_button, TOTAL):
    if proceed_button is None:
        if TOTAL == 0:
            messagebox.showinfo("Authentication Successful", "You are authenticated !")
            # # path = r"C:\Users\Mukul  Dev\OneDrive\Desktop\03 John Wick Chapter 3 Parabellum - Action 2019 Eng Subs 720p [H264-mp4].mp4"
            # # run_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
            # path = r"C:\Users\Mukul  Dev\OneDrive\Desktop\Markdown to Table\run.pyw"
            # run_path = r"C:\Users\Mukul  Dev\AppData\Local\Programs\Python\Python311\pythonw.exe"
            # subprocess.Popen([run_path, path])
            sys.exit()
