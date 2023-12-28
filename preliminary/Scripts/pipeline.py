import os
import numpy as np
import data_acquisition
import data_preprocessing
import artifact_rejection
import settings

def print_signal_summary(eeg_data, stage):
    print(f"--- {stage} Signal Summary ---")
    print(f"Shape: {eeg_data.shape}")
    print(f"Mean: {np.mean(eeg_data, axis=1)}")
    print(f"Standard Deviation: {np.std(eeg_data, axis=1)}\n")

def run_pipeline():
    # Step 1: Data Acquisition
    print("Running Data Acquisition...")
    eeg_data = data_acquisition.generate_placeholder_eeg_data()
    data_acquisition.save_data(eeg_data)
    print_signal_summary(eeg_data, "Data Acquisition")

    # Define the file path for the saved EEG data
    file_name = f"subject_{settings.subjID}_day_{settings.expDay}_eeg_data.npy"
    file_path = os.path.join(settings.subject_path_init(), f"subject_{settings.subjID}", file_name)

    # Load the saved EEG data
    eeg_data = np.load(file_path)

    # Step 2: Data Preprocessing
    print("Running Data Preprocessing...")
    preprocessed_data = data_preprocessing.preprocess_eeg_data(eeg_data)
    print_signal_summary(preprocessed_data, "Data Preprocessing")

    # Reshape preprocessed_data for artifact rejection
    reshaped_data = preprocessed_data.transpose(1, 0, 2).reshape(preprocessed_data.shape[1], -1)

    # Step 3: Artifact Rejection
    print("Running Artifact Rejection...")
    cleaned_data = artifact_rejection.artifact_rejection_pipeline(reshaped_data, reshaped_data)
    print_signal_summary(cleaned_data, "Artifact Rejection")

    # Here you can add further steps, like analysis, classification, etc.

    print("Pipeline execution completed.")

if __name__ == "__main__":
    run_pipeline()
