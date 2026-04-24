import numpy as np
import pycatch22
from pathlib import Path
from tqdm import tqdm
from scipy import stats, signal
from pycatch22 import catch22_all



# -----------------------------------
# Feature extraction functions
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