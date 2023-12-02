# Part 1 of the Pipeline

import numpy as np
import os

def generate_sample_data():
    # Create the data directory if it doesn't exist
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Number of trials, samples per trial, and channels
    n_trials = 100
    n_samples = 550
    n_channels = 32

    # Generate random EEG data
    eeg_data = np.random.randn(n_trials, n_samples, n_channels)

    # Channel names
    channels = ['P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'T8', 'F8', 'C4', 'F4', 'Fp2', 'Fz', 'C3', 'F3', 'Fp1', 'T7', 'F7', 'Oz', 'PO3', 'AF3', 'FC5', 'FC1', 'CP5', 'CP1', 'CP2', 'CP6', 'AF4', 'FC2', 'FC6', 'PO4']
    assert len(channels) == n_channels, "Channel count mismatch"

    # Generate binary category labels
    categories = np.random.randint(2, size=n_trials)

    # Save data
    np.save(os.path.join(data_dir, 'EEG_epochs_sample.npy'), eeg_data)
    np.save(os.path.join(data_dir, 'y_categories_sample.npy'), categories)

    print("Sample data generated and saved in './data/' directory.")

if __name__ == "__main__":
    generate_sample_data()
