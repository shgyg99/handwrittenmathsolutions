import os
import random
import shutil

def clean_folders(main_folder):
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        
        if os.path.isdir(subfolder_path):  # فقط پوشه‌ها را بررسی کن
            image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('png', 'jpg', 'jpeg'))]

            if len(image_files) > 300:  # اگر بیش از 300 عکس دارد
                print(f"Folder '{subfolder}' has {len(image_files)} images. Reducing to 300.")
                
                # انتخاب تصادفی 300 عکس برای نگه داشتن
                keep_images = set(random.sample(image_files, 300))

                # حذف سایر تصاویر
                for img in image_files:
                    if img not in keep_images:
                        img_path = os.path.join(subfolder_path, img)
                        os.remove(img_path)
                        print(f"Deleted: {img_path}")

                print(f"Reduced '{subfolder}' to 300 images.\n")

# انتخاب پوشه اصلی
main_folder = input("Enter the main folder path: ")
if os.path.exists(main_folder):
    clean_folders(main_folder)
    print("Cleaning process completed.")
else:
    print("Invalid folder path. Please check and try again.")
