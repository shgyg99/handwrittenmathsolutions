import csv
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import glob

image_paths = glob.glob('.\cropped\*')
data = []

def save_label():
    latex_code = entry.get()
    data.append([current_image, latex_code])
    load_next_image()

def load_next_image():
    global current_image, start_image_index, end_image_index
    if start_image_index < end_image_index:
        current_image = image_paths[start_image_index]
        start_image_index += 1

        img = Image.open(current_image)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        number.config(text=f"Image Index: {start_image_index}")
    else:
        save_to_csv()
        print("All images were labeled!")
        root.destroy()

def save_to_csv():
    with open(f"labels{start_index}-{end_image_index}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Path", "LaTeX Label"])
        writer.writerows(data)
    print("csv file saved!")

root = tk.Tk()
root.geometry("800x600")
start_image_index = int(input('enter start index: '))
start_index = start_image_index
end_image_index= int(input('enter end index: '))

number = tk.Label(root, text=f"Image Index: {start_image_index}")
number.pack()

panel = tk.Label(root)
panel.pack()

entry = tk.Entry(root, width=50, font=('Helvetica', 20))
entry.pack()

save_button = tk.Button(root, text="Next", command=save_label)
save_button.pack()

load_next_image()
root.mainloop()
