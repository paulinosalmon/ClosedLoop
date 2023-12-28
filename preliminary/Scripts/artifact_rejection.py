import numpy as np
import settings

def perform_ssp(X_training, X_noisy_test, variance_threshold):
    # Perform SVD on the training data
    U, S, V = np.linalg.svd(X_training, full_matrices=False)

    # Calculate the total variance
    total_variance = np.sum(S**2)

    # Determine the number of components to retain
    retained_variance = S**2 / total_variance
    retained_components = np.where(retained_variance >= variance_threshold)[0]

    if len(retained_components) == 0:
        print("No components retained. Adjusting to retain at least one component.")
        retained_components = [0]  # Retain at least the first component

    # Project out the retained components from the test data
    U_retained = U[:, retained_components]
    V_retained = V[retained_components, :]

    # Handle the case where only one component is retained
    if len(retained_components) == 1:
        singular_value_matrix = np.diag(S[retained_components])
        reconstructed_signal = U_retained @ singular_value_matrix @ V_retained
    else:
        reconstructed_signal = U_retained @ np.diag(S[retained_components]) @ V_retained.T

    # Subtract the reconstructed signal from the original noisy test data
    X_clean_test = X_noisy_test - reconstructed_signal

    return X_clean_test

def artifact_rejection_pipeline(X_training, X_noisy_test):
    """
    Pipeline for artifact rejection using SSP.

    :param X_training: Training set matrix [channels x time points]
    :param X_noisy_test: Test set matrix before denoising [channels x time points]
    :return: X_clean_test - Denoised test set matrix [channels x time points]
    """
    if settings.SSP:
        return perform_ssp(X_training, X_noisy_test, settings.thresholdSSP)
    else:
        return X_noisy_test  # Return the original data if SSP is not enabled

# The rest of the script remains the same


if __name__ == "__main__":
    # Example usage with simulated data for demonstration

    # Simulate some EEG data for training and testing
   pass
