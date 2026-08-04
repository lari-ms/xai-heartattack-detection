import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pickle

#baseline wander removal
#denoising
#normalization

#baseline wander removal
def remove_baseline_wander(ecg_signal, freq=100):
    """
    Removes baseline wander from a single ECG using
    Chazal's double median filter.

    Parameters:
        ecg_signal: ECG signal with shape (timesteps, leads)
        freq: Sampling frequency in Hz

    Returns:
        ECG signal after baseline correction.
    """

    filter_200ms = int(freq * 0.2)
    if filter_200ms % 2 == 0:
        filter_200ms += 1

    filter_600ms = int(freq * 0.6)
    if filter_600ms % 2 == 0:
        filter_600ms += 1

    baseline_corrected = np.empty_like(ecg_signal)

    for lead in range(ecg_signal.shape[1]):
        median_200 = signal.medfilt(ecg_signal[:, lead], filter_200ms)
        baseline = signal.medfilt(median_200, filter_600ms)

        baseline_corrected[:, lead] = ecg_signal[:, lead] - baseline

    return baseline_corrected

#denoising
def denoise_signal(ecg_signal, freq=100):
    """
    Removes high-frequency noise from a single ECG using
    a Butterworth low-pass filter.

    Parameters:
        ecg_signal: ECG signal with shape (timesteps, leads)
        freq: Sampling frequency in Hz

    Returns:
        Denoised ECG signal.
    """

    filtered_signal = ecg_signal.copy()

    b, a = signal.butter(
        N=12,
        Wn=35,
        btype="lowpass",
        fs=freq
    )

    for lead in range(filtered_signal.shape[1]):
        filtered_signal[:, lead] = signal.filtfilt(
            b,
            a,
            filtered_signal[:, lead]
        )

    return filtered_signal


#standardization
def standardize_signals(X_train, X_val, X_test, scaler_path=None):
    """
    Standardizes ECG signals using lead-wise Z-score normalization.

    The scaler is fitted only on the training set and then applied
    to the validation and test sets.

    Parameters:
        X_train: (n_train, timesteps, leads)
        X_val: (n_val, timesteps, leads)
        X_test: (n_test, timesteps, leads)
        scaler_path: Optional path to save the fitted scaler.

    Returns:
        X_train_scaled
        X_val_scaled
        X_test_scaled
        scaler
    """

    _, _, n_leads = X_train.shape

    X_train_reshaped = X_train.reshape(-1, n_leads)
    X_val_reshaped = X_val.reshape(-1, n_leads)
    X_test_reshaped = X_test.reshape(-1, n_leads)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    if scaler_path is not None:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

#preprocessing pipeline
def preprocess_ecg(ecg_signal):
    ecg = remove_baseline_wander(ecg_signal)
    ecg = denoise_signal(ecg)
    return ecg
