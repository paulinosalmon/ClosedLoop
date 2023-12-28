# feedback_generator.py
import numpy as np
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
import os
import settings
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def transfer_function(classifier_output):
    return 0.81 * sigmoid(classifier_output) + 0.17

def load_and_process_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((175, 175))
    img = ImageOps.equalize(img)
    return img

def create_composite_image(image1, image2, alpha):
    return Image.blend(image1, image2, alpha)

def get_random_image_path(category):
    base_dir = os.path.join(settings.base_dir_init(), 'imageStimuli', category)
    image_name = random.choice(os.listdir(base_dir))
    return os.path.join(base_dir, image_name)

def generate_feedback_image(classifier_outputs, category1, category2, label):
    alpha_values = [transfer_function(output) for output in classifier_outputs]
    averaged_alpha = np.mean(alpha_values[-3:])

    image_path1 = get_random_image_path(category1)
    image_path2 = get_random_image_path(category2)
    image1 = load_and_process_image(image_path1)
    image2 = load_and_process_image(image_path2)

    composite_image = create_composite_image(image1, image2, averaged_alpha)
    img = ImageTk.PhotoImage(composite_image)
    label.config(image=img)
    label.image = img

def run_gui(classifier_outputs, category1, category2):
    root = tk.Tk()
    root.title("EEG Feedback Display")
    label = tk.Label(root)
    label.pack()

    generate_feedback_image(classifier_outputs, category1, category2, label)
    root.mainloop()

if __name__ == "__main__":
    # Example usage
    classifier_outputs = [0.5, -0.3, 0.8]  # Placeholder classifier outputs
    run_gui(classifier_outputs, 'female', 'outdoor')
