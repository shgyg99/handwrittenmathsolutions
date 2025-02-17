import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling Tool")

        # Button to select the main folder
        self.select_folder_btn = tk.Button(root, text="Select Main Folder", command=self.select_main_folder)
        self.select_folder_btn.pack(pady=10)

        # Label to show selected folder
        self.folder_label = tk.Label(root, text="No folder selected", fg="red")
        self.folder_label.pack()

        # Display current folder name
        self.current_folder_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
        self.current_folder_label.pack(pady=10)

        # Entry for label input
        self.label_entry = tk.Entry(root, width=40)
        self.label_entry.pack()

        # Button to submit label
        self.submit_btn = tk.Button(root, text="Submit Label", command=self.save_label)
        self.submit_btn.pack(pady=10)

        # Variables
        self.main_folder = None
        self.subfolders = []
        self.current_folder = None
        self.image_paths = []
        self.data = []

    def select_main_folder(self):
        """ Allows the user to select the main folder. """
        self.main_folder = filedialog.askdirectory(title="Select the root folder containing subfolders")
        if self.main_folder:
            self.folder_label.config(text=f"Selected: {self.main_folder}", fg="green")
            self.subfolders = [f for f in os.listdir(self.main_folder) if os.path.isdir(os.path.join(self.main_folder, f))]
            if not self.subfolders:
                messagebox.showerror("Error", "No subfolders found!")
                return
            self.load_next_folder()

    def load_next_folder(self):
        """ Loads the next folder for labeling. """
        if not self.subfolders:
            messagebox.showinfo("Info", "No more folders to label.")
            self.save_csv()
            return

        self.current_folder = self.subfolders.pop(0)
        folder_path = os.path.join(self.main_folder, self.current_folder)

        # Collect all image paths in the current folder
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)
                            if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]

        if not self.image_paths:
            self.load_next_folder()  # Skip empty folders
            return

        self.current_folder_label.config(text=f"Folder: {self.current_folder}", fg="blue")

    def save_label(self):
        """ Saves the entered label and applies it to all images in the folder. """
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showwarning("Warning", "Please enter a label!")
            return

        for img_path in self.image_paths:
            self.data.append([img_path, label])

        self.label_entry.delete(0, tk.END)
        self.load_next_folder()

    def save_csv(self):
        """ Saves the labeled data to a CSV file. """
        if not self.data:
            messagebox.showinfo("Info", "No labeled data to save.")
            return
        
        df = pd.DataFrame(self.data, columns=["image_path", "label"])
        csv_filename = os.path.join(self.main_folder, "labeled_images.csv")
        df.to_csv(csv_filename, index=False)
        messagebox.showinfo("Success", f"CSV file saved: {csv_filename}")
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.mainloop()
