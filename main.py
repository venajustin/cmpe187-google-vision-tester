import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
from get_files import *
from img_display import *
from file_list import * 

root = tk.Tk()
root.title("Google Cloud Test application")
root.geometry("900x600")
root.resizable(True, True)

# Main layout: 3 rows, 2 columns
root.rowconfigure(0, weight=2)
root.rowconfigure(1, weight=3)
root.columnconfigure(0, weight=1)

top_frame = ttk.Frame(root)
top_frame.grid(row=0, column=0, sticky="nsew")
top_frame.columnconfigure(0, weight=1);
top_frame.columnconfigure(1, weight=0);
top_frame.columnconfigure(2, weight=1);
top_frame.rowconfigure(0, weight=1);




input_img_frame = img_display(top_frame, "Input Image")
input_img_frame.grid(row=0, column=0, sticky="nsew")


output_img_frame = img_display(top_frame, "Output Image")
output_img_frame.grid(row=0, column=2, sticky="nsew")


canvas = tk.Canvas(top_frame, width=35, bg="white")
canvas.grid(row=0, column=1, columnspan=1, sticky="ns")
canvas.create_line(5, 100, 30, 100, arrow=tk.LAST, width=5, fill="black")


bottom_frame = ttk.Frame(root)
bottom_frame.grid(row=1, column=0, sticky="nsew")
bottom_frame.columnconfigure(0, weight=1);
bottom_frame.columnconfigure(1, weight=1);
bottom_frame.rowconfigure(0, weight=1);

# Input list
file_list = get_files("./input");

def test_callback(val):
    print(f"test callback called with {val}")
def change_input_callback(val):
    set_photo(input_img_frame, "./input/" + val)

bottom_left = create_file_list(bottom_frame, file_list, change_input_callback)
bottom_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# Output list
bottom_right = create_file_list(bottom_frame, [f"Entry {i}" for i in range(1, 51)], test_callback)
bottom_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

root.mainloop()


