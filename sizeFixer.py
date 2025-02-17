import os
from PIL import Image, ImageOps

# Settings
TARGET_SIZE = (300, 300)  # Final size for the image
x1, y1, x2, y2 = 50, 50, 200, 200  # Crop coordinates (adjust as needed)

# Asking the user for the folder
folder = input("Please enter one of the folders (e.g., EnglishAlphabet, PersianNumbers, Symbols): ")

# Checking if the entered folder is valid
if folder not in ['EnglishAlphabet', 'PersianNumbers', 'Symbols']:
    print("The entered folder is not valid.")
else:
    # Accessing the folder
    folder_path = f"./{folder}"

    # Traversing the files in the folder
    for subdir, dirs, files in os.walk(folder_path):
        for idx, file in enumerate(files):
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Only image files
                image_path = os.path.join(subdir, file)
                
                # Opening the image
                image = Image.open(image_path).convert("L")  # Convert to grayscale
                
                # If PersianNumbers is selected, invert the image
                if folder == "PersianNumbers":
                    image = ImageOps.invert(image)  # Invert colors to get black text on white background
                
                # Cropping the image
                
                # Creating the thumbnail
                image.thumbnail(TARGET_SIZE)
                
                # Creating a white canvas to place the cropped image
                canvas = Image.new("L", TARGET_SIZE, "white")
                
                # Calculating the offset to center the cropped image on the canvas
                offset_x = (TARGET_SIZE[0] - image.width) // 2
                offset_y = (TARGET_SIZE[1] - image.height) // 2
                canvas.paste(image, (offset_x, offset_y))
                
                # Saving the image with the same name, replacing the original file
                canvas.save(image_path)
                
                # print(f"Image {file} has been successfully processed and replaced.")

