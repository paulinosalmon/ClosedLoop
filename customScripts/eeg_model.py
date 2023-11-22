import numpy as np
from sklearn.linear_model import LogisticRegression

def normalize_and_scale_eeg_data(raw_eeg):
    mean_eeg = 55
    std_dev_eeg = 20
    z_normalized_eeg = (raw_eeg - mean_eeg) / std_dev_eeg
    max_scale = 3
    scaled_eeg = max(-1, min(1, z_normalized_eeg / max_scale))
    return scaled_eeg

def generate_sample_data(num_samples=1000):
    eeg_data = np.random.uniform(10, 100, num_samples)
    eeg_data = np.array([normalize_and_scale_eeg_data(data) for data in eeg_data])
    labels = np.array([1 if data > 0 else 0 for data in eeg_data])
    eeg_data = eeg_data.reshape(-1, 1)
    return eeg_data, labels

def train_logistic_regression(eeg_data, labels):
    model = LogisticRegression()
    model.fit(eeg_data, labels)
    return model
