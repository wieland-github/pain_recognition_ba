import numpy as np
import pycatch22
from pathlib import Path
from tqdm import tqdm
from scipy import stats, signal
from pycatch22 import catch22_all
import neurokit2 as nk



# -----------------------------------
# Feature extraction catch22 + time-domain features
# -----------------------------------

def extract_catch22_features(signal):
    """
    Extract catch22 features
    """
    features = catch22_all(signal)
    return features['values'], features['names']

def time_domain_features(signal):
    """
    Extract time-domain features
    """
    mean = np.mean(signal)
    std = np.std(signal)
    min = np.min(signal)
    max = np.max(signal)
    rms = np.sqrt(np.mean(signal**2))
    skewness = stats.skew(signal)
    kurtosis = stats.kurtosis(signal)

    vals = [mean, std, min, max, rms, skewness, kurtosis]
    names = ["mean", "std", "min", "max", "rms", "skewness", "kurtosis"]

    return np.array(vals), names


def frequency_domain_features(signal, sr):
    """"
    Exctract frequency-domain features
    """

    pass


def extract_all_features_from_one_signal(signal):
    """
    Extract all features for one signal
    """
    extractors = [extract_catch22_features, time_domain_features]

    all_vals, all_names = [], []
    for extractor in extractors:
        vals, names = extractor(signal)
        all_vals.append(np.asarray(vals))
        all_names.extend(names)

    return np.concatenate(all_vals), all_names


def extract_features_dataset(X, sensor_names):
    """
    Extract all features
    """
    n_samples, _, n_sensors = X.shape

    probe_vals, base_names = extract_all_features_from_one_signal(np.asarray(X[0, :, 0], dtype=float))
    n_features_per_signal = len(probe_vals)

    feature_names = [
        f"{sensor}__{name}"
        for sensor in sensor_names
        for name in base_names
    ]

    X_features = np.empty(
        (n_samples, n_sensors * n_features_per_signal), dtype=float
    )

    for i in range(n_samples):
        row = []
        for s in range(n_sensors):
            signal = np.asarray(X[i, :, s], dtype=float)
            vals, _ = extract_all_features_from_one_signal(signal)
            row.append(vals)
        X_features[i] = np.concatenate(row)

    return X_features, feature_names


# -----------------------------------
# Feature extraction Neurokit2
# -----------------------------------

def extract_neurokit2_all_features_from_one_signal(signal, sensor_names, sr):
    """
    Extract features using Neurokit2
    """

    all_features = []

    # replace nan and inf with 0
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # extract features for each sensor
    for sensor in sensor_names:
        if sensor == "Bvp":
            processed, info =nk.ppg_process(signal, sampling_rate=sr)
            features = nk.ppg_analyze(processed, sampling_rate=sr, method="interval-related")
        elif sensor == "Eda_E4" or sensor == "Eda_RB":
            processed, info = nk.eda_process(signal, sampling_rate=sr)
            features = nk.eda_analyze(processed, sampling_rate=sr, method="interval-related")
        elif sensor == "Resp":
            processed, info = nk.rsp_process(signal, sampling_rate=sr)
            features = nk.rsp_analyze(processed, sampling_rate=sr, method="interval-related")
        elif sensor == "Ecg":
            processed, info = nk.ecg_process(signal, sampling_rate=sr)
            features = nk.ecg_analyze(processed, sampling_rate=sr, method="interval-related")
        elif sensor == "Emg":
            processed, info = nk.emg_process(signal, sampling_rate=sr)
            features = nk.emg_analyze(processed, sampling_rate=sr, method="interval-related")

        features = features.add_prefix(f"{sensor}__")
        all_features.append(features)

    return np.concat(all_features, axis=1)
