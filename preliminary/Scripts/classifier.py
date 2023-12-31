# classifier.py
import numpy as np
import time
import joblib
import os

from settings import classifier, config, config_score
from datetime import datetime
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.base import clone

def evaluate_model(classifier, X, y, cv=3):
    """
    Evaluate the classifier using cross-validation and return the average score.
    """
    scores = cross_val_score(clone(classifier), X, y, cv=cv)
    return np.mean(scores)

def evaluate_saved_model(test_X, test_y):
    """
    Evaluate the saved classifier model and return the mean classifier decoding error rate.

    Parameters:
    - test_X: Test features
    - test_y: True labels for the test data

    Returns:
    - Mean classifier decoding error rate
    """
    # Load the saved model
    model = joblib.load("../model/best_model.pkl")

    # Predict using the loaded model
    predicted_y = model.predict(test_X)

    # Calculate accuracy
    accuracy = accuracy_score(test_y, predicted_y)

    # Calculate and return error rate
    error_rate = 1 - accuracy
    return error_rate

def calculate_bias_offset(classifier, X, y, cv=3, limit=0.125):
    """
    Calculate the bias offset based on cross-validation predicted probabilities.
    """
    # Perform cross-validation and get the probabilities for the positive class
    cross_val_probs = cross_val_predict(clone(classifier), X, y, cv=cv, method='predict_proba')
    # Calculate the mean probability across folds for the positive class
    mean_proba = np.mean(cross_val_probs[:, 1])
    # Calculate the bias as the deviation from the chance level (0.5)
    bias = mean_proba - 0.5
    # Limit the bias to within the specified range
    bias_offset = np.clip(bias, -limit, limit)
    return bias_offset

def calculate_weighted_moving_average(data, alpha=0.5):
    """Calculate the weighted moving average of the data with an exponential decay."""
    wma = np.zeros_like(data)
    wma[0] = data[0]
    for i in range(1, len(data)):
        wma[i] = alpha * data[i] + (1 - alpha) * wma[i - 1]
    return wma

def calculate_classifier_output(probabilities, task_relevant_index):
    """
    Calculate the classifier output as the difference between
    the probability of the task-relevant category and the task-irrelevant category.
    """
    task_irrelevant_index = 1 - task_relevant_index
    classifier_output = probabilities[:, task_relevant_index] - probabilities[:, task_irrelevant_index]
    return classifier_output

def asymmetric_sigmoid_transfer(output, inflection_point=0.6, lower_bound=0.17, upper_bound=0.98):
    """
    Custom sigmoid function that maps the output to a specific range with an inflection point.
    
    Parameters:
    - output: Input value or array
    - inflection_point: The point at which the sigmoid curve has its midpoint
    - lower_bound: The lower bound of the output range
    - upper_bound: The upper bound of the output range
    
    Returns:
    - Sigmoid output mapped between lower_bound and upper_bound
    """
    sigmoid_range = upper_bound - lower_bound

    # Adjust the output to shift the sigmoid curve
    adjusted_output = output - inflection_point

    sigmoid_output = 1 / (1 + np.exp(-adjusted_output))
    return lower_bound + sigmoid_range * sigmoid_output


def run_classifier(queue_gui, queue_artifact_rejection, queue_classifier):
    global is_model_generated
    queue_gui.put(f"[{datetime.now()}] [Classifier] Initializing classifier...")
    model_path = "../model/best_model.pkl"
    past_probabilities = []
    feedback_bias_offsets = []

    while True:
        try:
            queue_gui.put(f"[{datetime.now()}] [Classifier] Waiting for data...")
            X_processed, y_processed = queue_artifact_rejection.get(block=True)

            # Convert list to NumPy array if necessary and transpose
            X_processed = np.array(X_processed).T if isinstance(X_processed, list) else X_processed.T
            y_processed = np.array(y_processed).flatten() if isinstance(y_processed, list) else y_processed.flatten()

            # Check dimensions and ensure they match
            if X_processed.shape[0] != y_processed.shape[0]:
                raise ValueError("Number of samples in X and y do not match")
            else:
                print(f"Model file already exists at {model_path}.")

                joblib.dump(classifier, model_path)
                print(f"Model saved at {model_path}")

            # Train the classifier
            classifier.fit(X_processed, y_processed)

            # ======================================================================== #

            if config["is_model_generated"] == False:
                print(f"[{datetime.now()}] [Classifier] Creating and saving a new model to {model_path}")
                joblib.dump(classifier, model_path)
                config["is_model_generated"] = True
        
            # Evaluate the model
            current_score = evaluate_model(classifier, X_processed, y_processed)
            queue_gui.put(f"[{datetime.now()}] [Classifier] Current model score: {current_score}")

            # Save the model if it performs better
            if current_score > config_score["best_score"]:
                config_score["best_score"] = current_score
                joblib.dump(classifier, model_path)
                queue_gui.put(f"[{datetime.now()}] [Classifier] New best model saved.")

            # ======================================================================== #

            queue_gui.put(f"[{datetime.now()}] [Classifier] Classifier trained.")

            # Estimate prediction probabilities
            probabilities = classifier.predict_proba(X_processed)
            past_probabilities.append(probabilities)

            # Calculate bias offset
            if len(past_probabilities) == 1:
                # For the first feedback run
                bias_offset = calculate_bias_offset(classifier, X_processed, y_processed)
            else:
                # For subsequent feedback runs, use the bias of the four most recent feedback blocks
                recent_probabilities = past_probabilities[-4:]
                recent_avg_proba = np.mean([probs[:, 1] for probs in recent_probabilities], axis=0)
                recent_bias = recent_avg_proba - 0.5
                bias_offset = np.clip(recent_bias, -0.125, 0.125)

            # Ensure bias_offset is a scalar before printing
            if not np.isscalar(bias_offset):
                bias_offset = np.mean(bias_offset)  # or bias_offset[0] if you want the first element

            queue_gui.put(f"[{datetime.now()}] [Classifier] Bias offset calculated: {bias_offset}")

            # ======================================================================== #

            # If more than one set of probabilities, calculate WMA
            if len(past_probabilities) > 1:
                proba_array = np.array(past_probabilities)
                wma_probabilities = np.apply_along_axis(calculate_weighted_moving_average, 0, proba_array)
                classifier_output = calculate_classifier_output(wma_probabilities[-1], task_relevant_index=1)
            else:
                classifier_output = calculate_classifier_output(probabilities, task_relevant_index=1)

            # Apply bias offset to the classifier output
            classifier_output -= bias_offset
            visibility_scores = np.array([asymmetric_sigmoid_transfer(output) for output in classifier_output])
            # ======================================================================== #

            # Calculate mean classifier error decoding rate

            error_rate = evaluate_saved_model(X_processed, y_processed)
            queue_gui.put(f"[{datetime.now()}] [Classifier] Best model's current mean classifier error decoding rate: {error_rate*100}%")

            # ======================================================================== #

            # Calculate and log the average visibility score
            average_visibility_score = np.mean(visibility_scores)
            queue_gui.put(f"[{datetime.now()}] [Classifier] Average visibility score: {average_visibility_score:.3f}")

            queue_gui.put(f"[{datetime.now()}] [Classifier] Data processing completed.")

            # Push the processed data and labels to the queue
            queue_classifier.put(average_visibility_score)
            message = f"[{datetime.now()}] [Classifier] Data pushed to queue: {average_visibility_score}"
            print(message)
            queue_gui.put(message)
            queue_gui.put(f"===================================")


        except Exception as e:
            queue_gui.put(f"[{datetime.now()}] [Classifier] Error encountered: {e}")
            break

# Note: Adjust the `task_relevant_index` if the task-relevant category
# is not represented by the second column (index 1) in the probabilities array.
