import numpy as np
import os
import time
import logging
import pickle
import tkinter as tk
from threading import Thread, Event

# Global variable for the feedback thread
feedback_thread = None

# Placeholder function for EEG data acquisition
def acquire_eeg_data(emotiv_device):
    # Placeholder: Simulate random EEG data acquisition
    # Replace with actual data acquisition logic
    return np.random.rand(32)  # Assuming 32 channels

# Placeholder function for EEG data preprocessing
def preprocess_eeg_data(eeg_data):
    # Placeholder: Basic preprocessing example
    # Replace with actual preprocessing steps (filtering, artifact removal, etc.)
    return eeg_data - np.mean(eeg_data, axis=0)  # Example: mean subtraction

# Placeholder function for feature extraction
def extract_features(preprocessed_data):
    # Placeholder: Basic feature extraction example
    # Replace with actual feature extraction logic
    return np.random.rand(160)  # Adjust the size to match the model's expected input

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def provide_feedback(prediction, canvas, circle_item):
    # Update the circle color based on the prediction
    color = '#ffcccc' if prediction == 0 else '#ccffcc'  # Light red for '0', light green for '1'
    canvas.itemconfig(circle_item, fill=color)

def setup_logging():
    log_dir = "./log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, 'real_time_feedback.log')),
                            logging.StreamHandler()
                        ])

def real_time_feedback(model, canvas, circle_item, stop_event):
    while not stop_event.is_set():
        new_eeg_data = acquire_eeg_data(None)
        preprocessed_data = preprocess_eeg_data(new_eeg_data)
        features = extract_features(preprocessed_data)

        # Reshape features to 2D array as expected by scikit-learn
        features_reshaped = features.reshape(1, -1)

        prediction = model.predict(features_reshaped)
        canvas.after(0, provide_feedback, prediction, canvas, circle_item)
        logging.info(f"Prediction: {prediction}, Feedback provided")
            
        # Wait for 1 second before the next iteration
        time.sleep(1)

def main():

    global feedback_thread  # Declare feedback_thread as global

    try:
        setup_logging()
        model_path = os.path.join('./model/', 'best_logistic_regression_model.pkl')
        model = load_model(model_path)

        root = tk.Tk()
        root.title("EEG Feedback System")

        # Create a canvas for drawing the circle
        canvas = tk.Canvas(root, width=200, height=200)
        canvas.pack()

        # Draw a circle on the canvas
        circle_item = canvas.create_oval(50, 50, 150, 150, fill="#f0f0f0")

        stop_event = Event()

        def start_feedback():
            global feedback_thread  # Use the global variable
            if feedback_thread is None or not feedback_thread.is_alive():
                stop_event.clear()
                feedback_thread = Thread(target=real_time_feedback, args=(model, canvas, circle_item, stop_event))
                feedback_thread.start()

        def stop_feedback():
            global feedback_thread  # Use the global variable
            stop_event.set()
            if feedback_thread is not None:
                feedback_thread.join()
                feedback_thread = None

        # Start and Stop buttons
        start_button = tk.Button(root, text="Start", command=start_feedback)
        start_button.pack(side=tk.LEFT, padx=10, pady=10)

        stop_button = tk.Button(root, text="Stop", command=stop_feedback)
        stop_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Exit Program button
        exit_button = tk.Button(root, text="Exit Program", command=root.destroy)
        exit_button.pack(side=tk.RIGHT, padx=10, pady=10)

        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()

    except KeyboardInterrupt:
        # Handle the Ctrl+C keyboard interrupt
        if stop_event is not None:
            stop_event.set()
        if feedback_thread is not None:
            feedback_thread.join()
        logging.info("Keyboard Interrupt received, shutting down.")

if __name__ == "__main__":
    main()
