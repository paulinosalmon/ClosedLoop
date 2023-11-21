import numpy as np
from eeg_model import normalize_and_scale_eeg_data, logistic_model

def classify_eeg_data(eeg_data):
    eeg_data = np.array([eeg_data]).reshape(-1, 1)
    prediction = logistic_model.predict(eeg_data)
    return 'good' if prediction[0] == 1 else 'bad'

def get_eeg_data():
    raw_eeg = np.random.uniform(10, 100)
    return normalize_and_scale_eeg_data(raw_eeg)

def eeg_to_color(eeg_value):
    if eeg_value >= 0:
        intensity = int(255 * (1 - eeg_value))
        return f'#{intensity:02x}ff{intensity:02x}'
    else:
        intensity = int(255 * (1 + eeg_value))
        return f'#ff{intensity:02x}{intensity:02x}'

# Example usage
eeg_value = get_eeg_data()
classification = classify_eeg_data(eeg_value)
color = eeg_to_color(eeg_value)
