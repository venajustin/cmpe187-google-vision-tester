import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk



def create_file_list(parent, items, callback):
    frame = ttk.Frame(parent)

    listbox = tk.Listbox(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
    listbox.config(yscrollcommand=scrollbar.set)

    for item in items:
        listbox.insert(tk.END, item)

    def double_click_handler(event):
        selected_indices = listbox.curselection()
        if selected_indices:
            index = selected_indices[0]
            value = listbox.get(index)
            callback(value)
            # print(f"Double-clicked on item at index {index}: {value}")
    listbox.bind("<Double-Button-1>", double_click_handler)

    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return frame

