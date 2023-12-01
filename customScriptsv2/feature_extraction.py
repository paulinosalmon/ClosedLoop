import numpy as np
import os
from scipy.signal import welch

def compute_psd(data, fs):
    # Define frequency bands
    bands = {'Delta': (1, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 100)}
    
    # Initialize dictionary to hold PSD values
    psd_features = {band: [] for band in bands}

    # Compute PSD for each trial and channel
    for trial in data:
        for channel in trial.T:
            f, Pxx = welch(channel, fs, nperseg=512)
            for band, (low, high) in bands.items():
                # Extract power in the specific band
                idx_band = np.logical_and(f >= low, f <= high)
                psd_band_power = np.sum(Pxx[idx_band])
                psd_features[band].append(psd_band_power)

    return psd_features

def extract_features():
    data_dir = "./data/"
    fs = 500  # Sampling frequency

    # Load preprocessed data
    eeg_data = np.load(os.path.join(data_dir, 'EEG_epochs_preprocessed.npy'))

    # Compute PSD features
    psd_features = compute_psd(eeg_data, fs)

    # Save features
    for band, features in psd_features.items():
        np.save(os.path.join(data_dir, f'features_{band}.npy'), features)

    print("Features extracted and saved.")

if __name__ == "__main__":
    extract_features()
