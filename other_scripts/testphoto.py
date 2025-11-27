
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk

root = tk.Tk()
root.title("Google Cloud Test application")
root.geometry("900x600")
root.resizable(True, True)

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

###########

img_w_label = ttk.Frame(root)
img_w_label.grid(row=0,column=0,sticky="nwes")
img_w_label.columnconfigure(0, weight=1);
img_w_label.rowconfigure(0, weight=0);
img_w_label.rowconfigure(1, weight=1);
    
img_label = tk.Label(img_w_label, text="Test", bg="white", fg="black")
img_label.grid(row=0, column=0, sticky="nwes")

frame = ttk.Frame(img_w_label)
frame.grid(row=1,column=0,sticky="nwes")

img = Image.open("./input/1.jpg").resize((400,200))
photo1 = ImageTk.PhotoImage(img)
input_image_locator = tk.Label(frame, image=photo1)
input_image_locator.pack()
input_image_locator.place(relx=.5, rely=.5,anchor='center')

####################
tk.mainloop()
