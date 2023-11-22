import threading
import time
# Import necessary functions from the other scripts
from eeg_model import train_logistic_regression, generate_sample_data
from eeg_classification import classify_eeg_data, get_eeg_data, eeg_to_color
from eeg_visualization import setup_gui, update_graph_and_stimulus

def main():
    # Example: Train the model with sample data
    sample_eeg_data, sample_labels = generate_sample_data()
    logistic_model = train_logistic_regression(sample_eeg_data, sample_labels)

    # Setup GUI
    root, canvas_stimulus, stimulus, ax, canvas = setup_gui(update_graph_and_stimulus)
    eeg_values = []

    def update_loop():
        while True:
            # Update the graph and stimulus using the EEG data processing, classification, and visualization functions
            eeg_data = get_eeg_data()
            classification = classify_eeg_data(eeg_data, logistic_model)  # Pass the model to the classification function
            color = eeg_to_color(eeg_data)
            eeg_values.append(eeg_data)
            update_graph_and_stimulus(eeg_values, ax, canvas, canvas_stimulus, stimulus, color, classification)
            time.sleep(1)

    thread = threading.Thread(target=update_loop)
    thread.daemon = True
    thread.start()
    root.mainloop()

if __name__ == "__main__":
    main()
