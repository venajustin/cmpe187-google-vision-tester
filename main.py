"""

Google Cloud Vision testing application

Prerequisites:
    install dependencies from requirements.txt
    configure google cloud account and project
        ensure the Google Cloud Vision API is enabled
    Install gcloud cli and authenticate to google account
    Ensure tkinter works on your machine with `python -m tkinter`

Run with: 
    windows:
        py main.py
    unix:
        python3 main.py

"""



import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
from vision_tester.get_files import *
from vision_tester.img_display import *
from vision_tester.file_list import * 
from vision_tester.log_list import *
from vision_tester.gvision_interface import *
from vision_tester.image_labeling import *
from vision_tester.filter import *

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


middle_frame = ttk.Frame(top_frame)
middle_frame.grid(row=0, column=1, sticky="nsew")
middle_frame.columnconfigure(0, weight=1)
middle_frame.rowconfigure(0, weight=2)
middle_frame.rowconfigure(1, weight=1)
middle_frame.rowconfigure(2, weight=1)
middle_frame.rowconfigure(3, weight=2)

placeholder1 = tk.Label(middle_frame, bg='white')
placeholder1.grid(row=0,column=0,sticky="nsew")
placeholder2 = tk.Label(middle_frame, bg="white")
placeholder2.grid(row=3,column=0,sticky="nsew")

canvas = tk.Canvas(middle_frame, width=35, bg="white")
canvas.grid(row=0, column=0, sticky="nsew")
canvas.create_line(5, 100, 30, 100, arrow=tk.LAST, width=5, fill="black")

objects_cache = {
        'objects':[],
        'raw_output_objects': []
        }

def send_button_press():
   raw_objects = localize_objects(input_img_frame.orig_img)
   objects_cache['raw_output_objects'] = raw_objects
   objects, text = filter_objects(raw_objects, get_filter_input())
   objects_cache['objects'] = objects
   image = label_img(objects,input_img_frame.orig_img)
   set_photo(output_img_frame, image)
   set_log_list(output_listbox, text)

def apply_filter(event = None):
   objects, text = filter_objects(objects_cache['raw_output_objects'], get_filter_input())
   objects_cache['objects'] = objects
   image = label_img(objects,input_img_frame.orig_img)
   set_photo(output_img_frame, image)
   set_log_list(output_listbox, text)


send_button = tk.Button(middle_frame, text="Send", command=send_button_press)
send_button.grid(row=2, column=0, sticky="nsew")

bottom_frame = ttk.Frame(root)
bottom_frame.grid(row=1, column=0, sticky="nsew")
bottom_frame.columnconfigure(0, weight=1);
bottom_frame.columnconfigure(1, weight=1);
bottom_frame.rowconfigure(0, weight=0);
bottom_frame.rowconfigure(1, weight=1);

# Input list
file_list = get_files("./input");

def change_input_callback(val):
    set_photo_by_path(input_img_frame, "./input/" + val)
def highlight_image_callback(val):
    image = label_img(objects_cache['objects'],input_img_frame.orig_img, val)
    set_photo(output_img_frame, image)

bottom_left = create_file_list(bottom_frame, file_list, change_input_callback)
bottom_left.grid(row=0, column=0, sticky="nsew", rowspan=2, padx=5, pady=5)


# Output list
bottom_right, output_listbox = create_log_list(bottom_frame, [], highlight_image_callback)
bottom_right.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)


def get_filter_input():
    input_val = filter_input.get()
    return input_val

# search input
filter_frame = ttk.Frame(bottom_frame)
filter_frame.grid(row=0, column=1, sticky="nsew")
filter_frame.rowconfigure(0, weight=1)
filter_frame.columnconfigure(0, weight=0)
filter_frame.columnconfigure(1, weight=3)
filter_frame.columnconfigure(2, weight=0)
filter_label = tk.Label(filter_frame, text="Filter: ")
filter_label.grid(row=0,column=0,sticky="nsew")
filter_input = tk.Entry(filter_frame )
filter_input.grid(row=0,column=1,sticky="nsew")
filter_input.bind('<Return>', apply_filter);
filter_button = tk.Button(filter_frame, text="Apply", command=apply_filter)
filter_button.grid(row=0, column=2, sticky="nsew")

root.mainloop()


