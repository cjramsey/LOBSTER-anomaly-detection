from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

def get_reconstruction_errors(model, X):
    '''
    Calculate reconstruction errors using mean squared error (MSE).

    Args:
        model (tf.keras.Model) Trained keras model.
        X (np.array): Sequences to reconstruct.
    
    Returns:
        errors (np.array): Array containing reconstruction errors.
    '''

    X_pred = model.predict(X)
    errors = np.mean(np.square(X - X_pred), axis=(1,2))
    return errors

def get_anomaly_scores(errors):
    '''Calculate anomaly scores.'''
    return errors / np.sum(errors)

def get_skew_kurtosis(reconstruction_errors):
    '''
    Calculate and print skewedness and kurtosis of reconstruction error distribution.
    Large positive skewewness indicates model predicts normal results well.
    Large excess kurtosis (distribution has long thin right tail) indicates model performs poorly
    on irregular sequences, hence we can identify anomalies.
    
    Args:
        reconstruction_error (np.array): Array of reconstruction errors.
    
    Returns:
        skew_ (float): Skewedness of reconstruction error distribution.
        kurtosis_ (float): Kurtosis of reconstruction error distribution.
    '''
    skew_ = skew(reconstruction_errors)
    kurtosis_ = kurtosis(reconstruction_errors)
    print(f"Skewedness: {skew_:.4f}")
    print(f"Excess Kurtosis: {kurtosis_ - 3:.4f}")
    return skew_, kurtosis_

def plot_error_distribution(errors, save_fig=False):
    '''
    Plot histogram of reconstruction error distribution.
    High positive skewedness and excess kurtosis suggest that model may be 
    detecting anomalous sequences of LOB data.

    Args:
        errors (np.array): Array of reconstruction errors.
        save_fig (bool): Whether to save figure or not.

    Return:
        None (NoneType)
    '''

    plt.hist(errors, bins=100)
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Count")

    if save_fig:
        base_dir = os.path.dirname(__file__)
        directory = "plots"
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_errors.png"
        file_path = os.path.join(base_dir, directory, file_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(file_path)

    plt.show()

def plot_anomalies(reconstruction_errors, threshold, save_fig=False):
    '''
    Plot reconstruction errors, highlighting anomalies above given threshold.

    Args:
        reconstruction_errors (np.array): Array of reconstruction errors.
        threshold (float): kth-percentile value in reconstruction errors.
        save_fig (bool): Whether to save figure or not.

    Return:
        None (NoneType)
    '''

    if threshold is None:
        threshold = np.percentile(reconstruction_errors, 99)

    anomalies = reconstruction_errors > threshold
    
    plt.figure(figsize=(14, 3))
    plt.plot(reconstruction_errors, label="Reconstruction Error")
    plt.scatter(np.where(anomalies)[0], reconstruction_errors[anomalies], color="red", label="Anomalies")
    plt.axhline(threshold, color="r", linestyle="--", label="Threshold")
    plt.title("Detected Anomalies")
    plt.xlabel("Sample")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_fig:
        base_dir = os.path.dirname(__file__)
        directory = "plots"
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_anomalies.png"
        file_path = os.path.join(base_dir, directory, file_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(file_path)

    plt.show()

    

