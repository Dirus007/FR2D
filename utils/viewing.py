import tkinter as tk
from tkinter import simpledialog, messagebox


def view_people(ADMIN_PASSWORD, window, data):
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