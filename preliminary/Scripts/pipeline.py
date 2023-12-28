# pipeline.py
import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import data_acquisition
import data_preprocessing
import artifact_rejection
import classifier
import feedback_generator
import evaluation
import settings
import threading

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

def print_signal_summary(eeg_data, stage):
    print(f"--- {stage} Signal Summary ---")
    print(f"Shape: {eeg_data.shape}")
    print(f"Mean: {np.mean(eeg_data, axis=1)}")
    print(f"Standard Deviation: {np.std(eeg_data, axis=1)}\n")

def run_pipeline(root, label):
    # Step 1: Data Acquisition
    print("Running Data Acquisition...")
    eeg_data = data_acquisition.generate_placeholder_eeg_data()
    data_acquisition.save_data(eeg_data)
    print_signal_summary(eeg_data, "Data Acquisition")

    file_path = os.path.join(settings.subject_path_init(), f"subject_{settings.subjID}", f"subject_{settings.subjID}_day_{settings.expDay}_eeg_data.npy")
    eeg_data = np.load(file_path)

    # Step 2: Data Preprocessing
    print("Running Data Preprocessing...")
    preprocessed_data = data_preprocessing.preprocess_eeg_data(eeg_data)
    print_signal_summary(preprocessed_data, "Data Preprocessing")

    reshaped_data = preprocessed_data.transpose(1, 0, 2).reshape(preprocessed_data.shape[1], -1)

    # Step 3: Artifact Rejection
    print("Running Artifact Rejection...")
    cleaned_data = artifact_rejection.artifact_rejection_pipeline(reshaped_data, reshaped_data)
    print_signal_summary(cleaned_data, "Artifact Rejection")

    # Classification and Evaluation
    print("Running Classification and Evaluation...")
    y_labels = np.random.randint(0, 2, cleaned_data.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(cleaned_data, y_labels, test_size=0.25)

    trained_classifier = classifier.train_classifier(X_train, y_train)
    bias_offset = classifier.compute_bias_offset(X_train, y_train)
    feedback_signals = classifier.test_classifier_realtime(trained_classifier, X_test, bias_offset)    # Capture predictions
    predictions = trained_classifier.predict(X_test)

    print("Feedback Signals:", feedback_signals)
    print("Number of Feedback Signals:", len(feedback_signals))

     # Generate Visual Feedback
    print("Generating Visual Feedback...")
    task_relevant_category = 'female'  # Example category

    # Create frames for image and graph in the GUI
    image_frame = tk.Frame(root)
    image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    graph_frame = tk.Frame(root)
    graph_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Create label for image
    image_label = tk.Label(image_frame)
    image_label.pack()

    # Update the GUI with feedback signals
    feedback_generator.update_gui(root, feedback_signals, task_relevant_category, 'outdoor', image_label, graph_frame)

    # Calculate and print classifier error rate
    print("=============== Evaluating Model ===============")
    classifier_error_rate = evaluation.calculate_error_rate(predictions, y_test)
    print(f"Classifier Error Rate: {classifier_error_rate}")
    rerr = evaluation.calculate_rerr(classifier_error_rate)
    print(f"Relative Error Rate Reduction (RERR): {rerr}")

    print("Pipeline execution completed.")

def run_pipeline_in_thread(root, label):
    thread = threading.Thread(target=run_pipeline, args=(root, label))
    thread.start()

def main():
    root = tk.Tk()
    root.title("EEG Feedback Display")
    label = tk.Label(root)
    label.pack()

    def on_close():
        plt.close('all')  # Close all Matplotlib figures
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    run_pipeline(root, label)
    root.mainloop()


if __name__ == "__main__":
    main()
