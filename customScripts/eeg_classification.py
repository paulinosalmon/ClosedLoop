import numpy as np
from eeg_model import normalize_and_scale_eeg_data  # Import the function from eeg_model.py

def classify_eeg_data(eeg_data, model):
    eeg_data = np.array([eeg_data]).reshape(-1, 1)
    prediction = model.predict(eeg_data)
    return 'good' if prediction[0] == 1 else 'bad'

def get_eeg_data():
    raw_eeg = np.random.uniform(10, 100)
    return normalize_and_scale_eeg_data(raw_eeg)  # Now this function is recognized

def eeg_to_color(eeg_value):
    if eeg_value >= 0:
        intensity = int(255 * (1 - eeg_value))
        return f'#{intensity:02x}ff{intensity:02x}'
    else:
        intensity = int(255 * (1 + eeg_value))
        return f'#ff{intensity:02x}{intensity:02x}'
