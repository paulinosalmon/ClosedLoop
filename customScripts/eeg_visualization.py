import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def setup_gui(update_function):
    root = tk.Tk()
    root.title("EEG Feedback")

    canvas_stimulus = tk.Canvas(root, width=200, height=200)
    canvas_stimulus.pack()
    stimulus = canvas_stimulus.create_oval(50, 50, 150, 150, fill="gray")

    graph_window = tk.Toplevel(root)
    graph_window.title("EEG Data Graph")
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    widget = canvas.get_tk_widget()
    widget.pack()

    return root, canvas_stimulus, stimulus, ax, canvas

def update_graph_and_stimulus(eeg_values, ax, canvas, canvas_stimulus, stimulus, color, classification):
    eeg_data = eeg_values[-1]  # Assuming the latest EEG data is the last item in the list
    print(f"EEG Value = {eeg_data:.2f} ({classification})")  # Print the EEG value and classification

    ax.clear()
    ax.plot(eeg_values, marker='o')
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('EEG Value')
    ax.set_title('Real-time EEG Data')
    ax.axhline(0, color='gray', linestyle='--')
    canvas.draw()

    canvas_stimulus.itemconfig(stimulus, fill=color)
