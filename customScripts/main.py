import threading
import time
import eeg_classification as ec  # Handles EEG data classification and preprocessing
import eeg_visualization as ev   # Handles EEG data visualization and GUI

def main():
    root, canvas_stimulus, stimulus, ax, canvas = ev.setup_gui(ev.update_graph_and_stimulus)
    eeg_values = []

    def update_loop():
        while True:
            # Update the graph and stimulus using the EEG data processing, classification, and visualization functions
            ev.update_graph_and_stimulus(eeg_values, ax, canvas, canvas_stimulus, stimulus, ec.get_eeg_data, ec.classify_eeg_data, ec.eeg_to_color)
            time.sleep(1)

    thread = threading.Thread(target=update_loop)
    thread.daemon = True
    thread.start()
    root.mainloop()

if __name__ == "__main__":
    main()
