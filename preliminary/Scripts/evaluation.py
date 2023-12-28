# evaluation.py
import numpy as np

def calculate_error_rate(predictions, true_labels):
    errors = predictions != true_labels
    error_rate = np.mean(errors)
    return error_rate

def calculate_rerr(error_rate):
    return (1 - error_rate) / error_rate

# Example usage
if __name__ == "__main__":
    # Load predictions and true labels
    # predictions, true_labels = load_data()

    # Calculate error rate
    # error_rate = calculate_error_rate(predictions, true_labels)
    # print(f"Error Rate: {error_rate}")

    # Calculate RERR
    # rerr = calculate_rerr(error_rate)
    # print(f"RERR: {rerr}")
    pass
    