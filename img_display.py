
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk

def img_display(parent, title):
    
    img_w_label = ttk.Frame(parent)
    img_w_label.grid(row=0,column=0,sticky="nwes")
    img_w_label.columnconfigure(0, weight=1);
    img_w_label.rowconfigure(0, weight=0);
    img_w_label.rowconfigure(1, weight=1);
        
    img_label = tk.Label(img_w_label, text=title, bg="white", fg="black")
    img_label.grid(row=0, column=0, sticky="nwes")

    frame = ttk.Frame(img_w_label)
    frame.grid(row=1,column=0,sticky="nwes")

    img = Image.open("./resources/placeholder.png")
    photo1 = ImageTk.PhotoImage(img)
    input_image_locator = tk.Label(frame, image=photo1)
    input_image_locator.place(relx=.5, rely=.5,anchor='center')

    ## stop imgage object from being garbage collected 
    img_w_label.orig_img = img
    img_w_label.photo = photo1
    img_w_label.label = input_image_locator
    img_w_label.frame = frame

    def resize_image(event):
        w, h = event.width, event.height
        resized = img_w_label.orig_img.resize((w, h), Image.LANCZOS)
        img_w_label.photo = ImageTk.PhotoImage(resized)
        img_w_label.label.config(image=img_w_label.photo)

    frame.bind("<Configure>", resize_image)

    return img_w_label


def set_photo(imgframe, path):

    img = Image.open(path)
    scaled = img.resize((imgframe.frame.winfo_width(), imgframe.frame.winfo_height()), Image.LANCZOS)
    photo = ImageTk.PhotoImage(scaled)
    imgframe.label.config(image=photo)
    imgframe.photo = photo
    imgframe.orig_img = img
