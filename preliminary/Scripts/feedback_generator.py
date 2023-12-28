import numpy as np
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
import os
import settings
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

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

def update_gui(root, classifier_outputs, category1, category2, image_label, graph_frame):
    alpha_values = [transfer_function(output) for output in classifier_outputs]
    averaged_alpha = np.mean(alpha_values[-3:])

    image_path1 = get_random_image_path(category1)
    image_path2 = get_random_image_path(category2)
    image1 = load_and_process_image(image_path1)
    image2 = load_and_process_image(image_path2)

    composite_image = create_composite_image(image1, image2, averaged_alpha)
    img = ImageTk.PhotoImage(composite_image)
    image_label.config(image=img)
    image_label.image = img

    def update_plot():
        plt.figure(figsize=(6, 4))
        plt.plot(classifier_outputs, label='Real-Time Category Decoding')
        plt.ylim([-1, 1])
        plt.xlabel('Trial Number')
        plt.ylabel('Real-Time Category Decoding')
        plt.legend()

        # Clear the previous graph and draw a new one
        for widget in graph_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(plt.gcf(), master=graph_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    # Schedule the plot update
    root.after(0, update_plot)

def run_gui(classifier_outputs, category1, category2):
    root = tk.Tk()
    root.title("EEG Feedback Display")

    # Create frames for image and graph
    image_frame = tk.Frame(root)
    image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    graph_frame = tk.Frame(root)
    graph_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Create label for image
    image_label = tk.Label(image_frame)
    image_label.pack()

    # Initial update of GUI components
    update_gui(classifier_outputs, category1, category2, image_label, graph_frame)

    def on_close():
        print("Closing GUI...")
        root.quit()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()

if __name__ == "__main__":
    classifier_outputs = [0.5, -0.3, 0.8]  # Placeholder classifier outputs
    run_gui(classifier_outputs, 'female', 'outdoor')
