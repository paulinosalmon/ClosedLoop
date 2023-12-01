import numpy as np
import os
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=1)
    return y

def remove_artifacts(eeg_data, threshold=100):
    # Simple threshold-based artifact removal
    std_per_trial = np.std(eeg_data, axis=1)
    clean_data = eeg_data[std_per_trial.max(axis=1) < threshold, :, :]
    return clean_data

def preprocess_eeg_data():
    data_dir = "./data/"
    fs = 500  # Sampling frequency

    # Load data
    eeg_data = np.load(os.path.join(data_dir, 'EEG_epochs_sample.npy'))
    categories = np.load(os.path.join(data_dir, 'y_categories_sample.npy'))

    # Bandpass filter settings
    lowcut = 1.0
    highcut = 40.0

    # Apply bandpass filter
    eeg_data_filtered = bandpass_filter(eeg_data, lowcut, highcut, fs, order=6)

    # Remove artifacts
    eeg_data_clean = remove_artifacts(eeg_data_filtered)

    # Save preprocessed data
    np.save(os.path.join(data_dir, 'EEG_epochs_preprocessed.npy'), eeg_data_clean)

    print("EEG data preprocessed and saved.")

if __name__ == "__main__":
    preprocess_eeg_data()
