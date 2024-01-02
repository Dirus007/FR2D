import pickle
from tkinter import simpledialog, messagebox


def delete_people(data, encodings_file, ADMIN_PASSWORD, window):
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