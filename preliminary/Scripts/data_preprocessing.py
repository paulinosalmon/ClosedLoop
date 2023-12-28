# data_preprocessing.py
import numpy as np
import mne
import settings

def preprocess_eeg_data(eeg_data):
    sfreq = settings.samplingRate
    event_id = 1
    tmin = -0.1
    tmax = 0.8

    ch_names = settings.channelNames
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Transpose eeg_data to match MNE structure (channels x samples)
    eeg_data = eeg_data.T

    # Linear detrending
    eeg_data = mne.filter.detrend(eeg_data, axis=1)

    raw = mne.io.RawArray(eeg_data, info)
    raw.filter(None, 40, fir_design='firwin', phase='zero-double')
    raw.resample(settings.samplingRateResample)

    n_samples = eeg_data.shape[1]
    events = np.array([[i, 0, event_id] for i in range(n_samples)])

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)
    epochs.apply_baseline(baseline=(None, 0))

    # Manual Z-scoring (standardizing)
    epochs_data = epochs.get_data()
    epochs_data = (epochs_data - epochs_data.mean(axis=2, keepdims=True)) / epochs_data.std(axis=2, keepdims=True)

    return epochs_data

def split_data_for_loro(eeg_data, labels, num_runs=settings.numRuns, blocks_per_run=settings.numBlocks):
    trials_per_block = len(eeg_data) // (num_runs * blocks_per_run)
    splits = []

    for run in range(num_runs):
        start_idx = run * blocks_per_run * trials_per_block
        end_idx = start_idx + blocks_per_run * trials_per_block

        # Check if indices are within the bounds of the dataset
        if start_idx >= len(eeg_data) or end_idx > len(eeg_data):
            print(f"Skipping run {run}: start_idx={start_idx}, end_idx={end_idx}, data_length={len(eeg_data)}")
            continue

        # Ensure we have data for the test set
        if end_idx - start_idx <= 0:
            print(f"No data for run {run}: start_idx={start_idx}, end_idx={end_idx}")
            continue

        X_test = eeg_data[start_idx:end_idx]
        y_test = labels[start_idx:end_idx]
        X_train = np.concatenate([eeg_data[:start_idx], eeg_data[end_idx:]])
        y_train = np.concatenate([labels[:start_idx], labels[end_idx:]])

        print(f"Run {run}: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        splits.append((X_train, X_test, y_train, y_test))

    return splits

if __name__ == "__main__":
    file_path = '../subjectsData/subject_00/subject_00_day_1_eeg_data.npy'  # Update with the actual path
    eeg_data = np.load(file_path)
    preprocessed_data = preprocess_eeg_data(eeg_data)


