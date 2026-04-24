import numpy as np
import pandas as pd
from pathlib import Path


#-----------------------------------
# Feature engineering functions
#-----------------------------------

def subject_baseline_normalization(X, y, subjects, baseline_class):
    """
    Normalize features by subtracting the subject-specific baseline
    """

    X_normalized = X.copy()

    for subject in np.unique(subjects):
        subject_mask = subjects == subject 
        baseline_mask = subject_mask & (y == baseline_class)
        baseline = X[baseline_mask].mean(axis=0)
        X_normalized[subject_mask] -= baseline

    return X_normalized