import threading
import time
import eeg_data_preprocessing as edp
import eeg_visualization as ev

def main():
    root, canvas_stimulus, stimulus, ax, canvas = ev.setup_gui(ev.update_graph_and_stimulus)
    eeg_values = []

    def update_loop():
        while True:
            ev.update_graph_and_stimulus(eeg_values, ax, canvas, canvas_stimulus, stimulus, edp.get_eeg_data, edp.classify_eeg_data, edp.eeg_to_color)
            time.sleep(1)

    thread = threading.Thread(target=update_loop)
    thread.daemon = True
    thread.start()
    root.mainloop()

if __name__ == "__main__":
    main()
