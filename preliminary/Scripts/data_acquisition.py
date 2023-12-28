# data_acquisition.py
import os
import numpy as np
import settings

def generate_placeholder_eeg_data():
    num_channels = 14
    num_samples = settings.numRuns * settings.numBlocks * settings.blockLen
    simulated_eeg = np.random.rand(num_samples, num_channels)
    return simulated_eeg

def save_data(eeg_data):
    subject_dir = os.path.join(settings.subject_path_init(), f"subject_{settings.subjID}")
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)

    file_name = f"subject_{settings.subjID}_day_{settings.expDay}_eeg_data.npy"
    file_path = os.path.join(subject_dir, file_name)
    np.save(file_path, np.array(eeg_data))
    print(f"Data saved to {file_path}")

# def acquire_data():
#     print("Looking for an EEG stream...")
#     streams = resolve_stream('type', 'EEG')

#     # Create an inlet to read from the stream
#     inlet = StreamInlet(streams[0])
#     print("Stream found. Starting data acquisition...")

#     # Initialize data buffer
#     eeg_data = []

#     # Data acquisition loop
#     for _ in range(settings.numRuns * settings.numBlocks * settings.blockLen):
#         sample, timestamp = inlet.pull_sample()
#         eeg_data.append(sample)

#         # Implement any real-time processing here if needed

#     return eeg_data

# def save_data(eeg_data):
#     # Create a directory for the subject if it doesn't exist
#     subject_dir = os.path.join(settings.subject_path_init(), f"subject_{settings.subjID}")
#     if not os.path.exists(subject_dir):
#         os.makedirs(subject_dir)

#     # Define the file name
#     file_name = f"subject_{settings.subjID}_day_{settings.expDay}_eeg_data.npy"
#     file_path = os.path.join(subject_dir, file_name)

#     # Save the data
#     np.save(file_path, np.array(eeg_data))
#     print(f"Data saved to {file_path}")

if __name__ == "__main__":
    eeg_data = generate_placeholder_eeg_data()
    save_data(eeg_data)
    