import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

def train_and_evaluate_model(features, labels, random_state):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_state)

    # Initialize logistic regression model
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    # Predict on the test set and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy, conf_matrix

def load_features(data_dir, bands, n_trials, n_channels):
    # Load features for specified bands and concatenate them
    features_list = []
    for band in bands:
        features_path = os.path.join(data_dir, f'features_{band}.npy')
        if os.path.exists(features_path):
            features = np.load(features_path)
            # Reshape the features to [n_trials, n_channels * n_features_per_channel]
            features = features.reshape(n_trials, n_channels * -1)
            features_list.append(features)
        else:
            logging.warning(f"Features for {band} band not found.")
    return np.concatenate(features_list, axis=1) if features_list else None

def main():
    data_dir = "./data/"
    model_dir = "./model/"
    log_dir = "./log/"
    num_iterations = 5  # Number of times to train the model with different splits
    best_accuracy = 0
    best_model = None

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, 'model_training.log')),
                            logging.StreamHandler()  # This will print to console
                        ])

    # Specify the bands to use (can be adjusted)
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    n_trials = 100  # Update this based on your dataset
    n_channels = 32  # Update this based on your dataset

    # Load features and labels
    features = load_features(data_dir, bands, n_trials, n_channels)
    labels = np.load(os.path.join(data_dir, 'y_categories_sample.npy'))

    for i in range(num_iterations):
        model, accuracy, conf_matrix = train_and_evaluate_model(features, labels, random_state=i)
        logging.info(f"Iteration {i}: Test Accuracy = {accuracy}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Save the best model
    if best_model is not None:
        model_path = os.path.join(model_dir, 'best_logistic_regression_model.pkl')
        with open(model_path, 'wb') as f:
            np.save(f, best_model)
        logging.info(f"Best model saved with accuracy: {best_accuracy}")

if __name__ == "__main__":
    main()
