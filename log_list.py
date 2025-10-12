
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk



def create_log_list(parent, items):
    frame = ttk.Frame(parent)

    listbox = tk.Listbox(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
    listbox.config(yscrollcommand=scrollbar.set)

    for item in items:
        listbox.insert(tk.END, item)

    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return frame, listbox

def set_log_list(listbox, items):

    listbox.delete(0, tk.END)
    for item in items:
        listbox.insert(tk.END, item)

    listbox.pack(side="left", fill="both", expand=True)
