# classifier.py
import numpy as np
import settings
from sklearn.model_selection import train_test_split, cross_val_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to train the classifier
def train_classifier(X_train, y_train):
    logging.info("Training the classifier...")
    classifier = settings.classifier
    classifier.fit(X_train, y_train)
    logging.info("Classifier trained successfully.")
    return classifier

# Function to compute bias offset
def compute_bias_offset(X_train, y_train):
    logging.info("Computing bias offset...")
    scores = cross_val_score(settings.classifier, X_train, y_train, cv=3)
    bias_offset = np.clip(np.mean(scores) - 0.5, -0.125, 0.125)
    logging.info(f"Bias offset computed: {bias_offset}")
    return bias_offset

# Function to test the classifier in real-time
def test_classifier_realtime(classifier, X_test, bias_offset):
    logging.info("Testing classifier in real-time...")
    feedback_signals = []
    for epoch in X_test:
        # Preprocess and artifact correct the epoch if necessary
        # For now, assuming epoch is ready for classification

        # Classify the epoch
        pc = classifier.predict_proba(epoch.reshape(1, -1))[0, 1]
        classifier_output = pc - (1 - pc) + bias_offset

        # Generate feedback signal
        feedback_signal = classifier_output
        feedback_signals.append(feedback_signal)

    logging.info("Real-time testing completed.")
    return feedback_signals

# Function to reshape EEG data for classification
def reshape_eeg_data(eeg_data):
    logging.info("Reshaping EEG data for classification...")
    reshaped_data = eeg_data.reshape(eeg_data.shape[0], -1)
    logging.info(f"Data reshaped to {reshaped_data.shape}")
    return reshaped_data

def loro_cross_validation(data_splits):
    error_rates = []
    for X_train, X_test, y_train, y_test in data_splits:
        trained_classifier = train_classifier(X_train, y_train)
        predictions = trained_classifier.predict(X_test)
        error_rate = 1 - accuracy_score(y_test, predictions)
        error_rates.append(error_rate)
    return np.mean(error_rates)

# Function to compute exponentially weighted moving average
def compute_ewma(data, alpha=0.5):
    logging.info("Computing exponentially weighted moving average...")
    ewma = np.zeros_like(data)
    ewma[0] = data[0]
    for i in range(1, len(data)):
        ewma[i] = alpha * data[i] + (1 - alpha) * ewma[i - 1]
    logging.info("EWMA computation completed.")
    return ewma
