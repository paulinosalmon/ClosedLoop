import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# Placeholder function for a machine learning model
def classify_eeg_data(eeg_data):
    return 'good' if eeg_data > 0.5 else 'bad'

# Function to simulate EEG data
def get_eeg_data():
    # Simulate raw EEG data between 10 to 100 microvolts
    raw_eeg = np.random.uniform(10, 100)

    # Improved normalization
    # Assuming the expected mean is 55 (midpoint of 10 to 100)
    # and a standard deviation that makes sense for this range, say 20
    mean_eeg = 55
    std_dev_eeg = 20

    # Z-score normalization
    z_normalized_eeg = (raw_eeg - mean_eeg) / std_dev_eeg

    # Scale Z-score to -1 to 1 range
    # Assuming most values fall within 3 standard deviations
    # This will clip outliers beyond 3 standard deviations
    max_scale = 3
    scaled_eeg = max(-1, min(1, z_normalized_eeg / max_scale))

    return scaled_eeg


# Function to convert EEG value to RGB color
def eeg_to_color(eeg_value):
    if eeg_value >= 0:
        # Green spectrum (from light green to green)
        intensity = int(255 * (1 - eeg_value))
        return f'#{intensity:02x}ff{intensity:02x}'
    else:
        # Red spectrum (from light red to red)
        intensity = int(255 * (1 + eeg_value))
        return f'#ff{intensity:02x}{intensity:02x}'

# Function to update the graph and stimulus
def update_graph_and_stimulus():
    while True:
        eeg_data = get_eeg_data()
        classification = classify_eeg_data(eeg_data)
        eeg_values.append(eeg_data)
        print(f"EEG Value = {eeg_data:.2f} ({classification})")

        # Update the graph
        ax.clear()
        ax.plot(eeg_values, marker='o')
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('EEG Value')
        ax.set_title('Real-time EEG Data')
        ax.axhline(0, color='gray', linestyle='--')
        canvas.draw()

        # Update the stimulus based on EEG data
        color = eeg_to_color(eeg_data)
        canvas_stimulus.itemconfig(stimulus, fill=color)

        time.sleep(1)  # Adjust the sleep time as needed

# Set up the main window
root = tk.Tk()
root.title("EEG Feedback")

# Set up the stimulus canvas
canvas_stimulus = tk.Canvas(root, width=200, height=200)
canvas_stimulus.pack()
stimulus = canvas_stimulus.create_oval(50, 50, 150, 150, fill="gray")

# Set up the graph in a separate window
graph_window = tk.Toplevel(root)
graph_window.title("EEG Data Graph")
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=graph_window)
widget = canvas.get_tk_widget()
widget.pack()

eeg_values = []

# Start the thread for updating the graph and stimulus
thread = threading.Thread(target=update_graph_and_stimulus)
thread.daemon = True  # Daemon thread will close when main window is closed
thread.start()

root.mainloop()
