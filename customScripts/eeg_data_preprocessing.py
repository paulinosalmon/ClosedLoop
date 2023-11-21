import numpy as np

def classify_eeg_data(eeg_data):
    return 'good' if eeg_data > 0.5 else 'bad'

def get_eeg_data():
    raw_eeg = np.random.uniform(10, 100)
    mean_eeg = 55
    std_dev_eeg = 20
    z_normalized_eeg = (raw_eeg - mean_eeg) / std_dev_eeg
    max_scale = 3
    scaled_eeg = max(-1, min(1, z_normalized_eeg / max_scale))
    return scaled_eeg

def eeg_to_color(eeg_value):
    if eeg_value >= 0:
        intensity = int(255 * (1 - eeg_value))
        return f'#{intensity:02x}ff{intensity:02x}'
    else:
        intensity = int(255 * (1 + eeg_value))
        return f'#ff{intensity:02x}{intensity:02x}'
