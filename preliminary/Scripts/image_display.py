import os
import random
from tkinter import Label, Tk, PhotoImage
from PIL import Image, ImageTk

def update_image_label(image_label, base_path='../imageStimuli'):
    categories = ['female', 'indoor', 'male', 'outdoor']
    selected_category = random.choice(categories)
    category_path = os.path.join(base_path, selected_category)

    if os.path.exists(category_path) and os.listdir(category_path):
        image_name = random.choice(os.listdir(category_path))
        image_path = os.path.join(category_path, image_name)
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference.
    else:
        print(f"No images found in {category_path}")

def display_random_image(window, image_label, base_path='../imageStimuli'):
    categories = ['female', 'indoor', 'male', 'outdoor']
    selected_category = random.choice(categories)
    category_path = os.path.join(base_path, selected_category)

    # Ensure the directory exists and has images
    if os.path.exists(category_path) and os.listdir(category_path):
        image_name = random.choice(os.listdir(category_path))
        image_path = os.path.join(category_path, image_name)

        # Load and display the image
        photo = PhotoImage(file=image_path)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to prevent garbage collection
    else:
        print(f"No images found in {category_path}")

def run_image_display():
    root = Tk()
    root.title("Random Image Display")

    # Label for the image
    label_image = Label(root)
    label_image.pack()

    # Display a random image on startup
    display_random_image(root, label_image)

    # Call the display function every 5000 ms (5 seconds)
    root.after(5000, lambda: display_random_image(root, label_image))

    root.mainloop()

if __name__ == "__main__":
    run_image_display()
