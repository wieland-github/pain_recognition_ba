import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, get_scorer, confusion_matrix
from sklearn.inspection import permutation_importance
from scipy.special import softmax

#-----------------------------------
# Define constants
#-----------------------------------
INNER_NSPLITS = 5
RANDOM_STATE = 42


# -----------------------------------
# Helper funktions
# -----------------------------------

def save_fold_results(fold_records, csv_path):
    """
    safe per fold results to csv
    """
    df = pd.DataFrame(fold_records)
    df.to_csv(csv_path, index=False)
    print(f"Per-fold results saved to {csv_path} ({len(df)} folds)")

def safe_results_binary(model_name, personalization, metrics):
    """
    safe the reults
    """

    results = {
    'model': model_name,
    'personalization': personalization,
    'k': metrics.get('k', None),
    'accuracy': metrics.get('accuracy'),
    'accuracy_std': metrics.get('accuracy_std'),
    'f1': metrics.get('f1'),
    'f1_std': metrics.get('f1_std'),
    'auc': metrics.get('auc'),
    'auc_std': metrics.get('auc_std'),
    'y_true': metrics.get('y_true'),
    'y_pred': metrics.get('y_pred'),
    'y_score': metrics.get('y_score'),
}

    # filename
    filename = f"results/models/{model_name}_{personalization}.csv"

    # safe
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")




def extract_signal_from_featurename(featurename):
    """
    Extract signal from feature name.
    bsp. "ECG__mean" -> "ECG"
    """
    if isinstance(featurename, str):
        return featurename.split("__")[0]

    return np.array([str(name).split("__")[0] for name in featurename])

def make_signal_groups(feature_names):
    """
    Make singal groups from feature names.
    """

    feature_names = np.asarray(feature_names)
    signals = extract_signal_from_featurename(feature_names)

    signal_groups = {}

    for signal in np.unique(signals):
        signal_groups[signal] = np.where(signals == signal)[0]

    return signal_groups


# -----------------------------------
# Define functions for model training
# -----------------------------------

def loso_binary_nested_cv(X, y, groups, model, space, model_type, csv_path_folds=None):
    """
    Loso for binary classification
    """

    outer_cv = LeaveOneGroupOut()
    inner_cv = GroupKFold(n_splits=INNER_NSPLITS)

    fold_acc, fold_f1, fold_auc = [], [], []
    fold_sens, fold_spec = [], []
    y_true_all, y_pred_all, y_score_all = [], [], []
    fold_records = []

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = groups[train_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        # inner CV for hyperparameter tuning
        # define the grid search with inner CV
        search = GridSearchCV(
            estimator = model,
            param_grid = space,
            cv = inner_cv.split(X_train_scaled, y_train, g_train),
            scoring = 'accuracy' if model_type == 'classifier'
                else 'neg_mean_squared_error',
            refit=True,
            n_jobs=-1,
        )

        # execute the grid search
        result = search.fit(X_train_scaled, y_train)

        # get the best model from the grid search
        best_model = result.best_estimator_

        # evaluate the best model on the test set
        if model_type == 'classifier':
            y_pred = best_model.predict(X_test_scaled)
            y_score = best_model.predict_proba(X_test_scaled)[:, 1]
        if model_type == 'regressor':
            y_score = best_model.predict(X_test_scaled)
            y_pred = np.clip(np.round(y_score), 0, 1).astype(int)

        # AUC needs both classes to be present
        if len(np.unique(y_test)) < 2:
            continue

        #store the results
        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_f1.append(f1_score(y_test, y_pred, zero_division=0))
        fold_auc.append(roc_auc_score(y_test, y_score))

        # sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        fold_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fold_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

        # store per-subject results for statistical testing
        fold_records.append({
            'subject': groups[test_idx][0],
            'accuracy': fold_acc[-1],
            'f1': fold_f1[-1],
            'auc': fold_auc[-1],
            'sensitivity': fold_sens[-1],
            'specificity': fold_spec[-1],
        })

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_score_all.append(y_score)

    # save the per-fold results
    save_fold_results(fold_records, csv_path_folds)

    # return the results
    return {
        'accuracy': np.mean(fold_acc),
        'f1': np.mean(fold_f1),
        'auc': np.mean(fold_auc),
        'sensitivity': np.mean(fold_sens),
        'specificity': np.mean(fold_spec),
        'accuracy_std': np.std(fold_acc),
        'f1_std': np.std(fold_f1),
        'auc_std': np.std(fold_auc),
        'sensitivity_std': np.std(fold_sens),
        'specificity_std': np.std(fold_spec),
        'y_true': np.concatenate(y_true_all),
        'y_pred': np.concatenate(y_pred_all),
        'y_score': np.concatenate(y_score_all),
        'fold_results': pd.DataFrame(fold_records),
    }

def loso_binary_baseline_check_nested_cv(X, y, groups, model, space, k_baseline, model_type, csv_path_folds=None):
    """
    Loso for binary classification
    """

    outer_cv = LeaveOneGroupOut()
    inner_cv = GroupKFold(n_splits=INNER_NSPLITS)
    rng = np.random.RandomState(RANDOM_STATE)

    fold_acc, fold_f1, fold_auc = [], [], []
    fold_sens, fold_spec = [], []
    y_true_all, y_pred_all, y_score_all = [], [], []
    fold_records = []

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = groups[train_idx]
        test_subject = groups[test_idx][0]

        # use all baseline samples from train subjects
        X_train_norm = X_train.copy()
        for subject in np.unique(g_train):
            subject_mask = g_train == subject
            baseline_mask = subject_mask & (y_train == 0)
            if np.sum(baseline_mask) > 0:
                X_train_norm[subject_mask] -= X_train[baseline_mask].mean(axis=0)

        # only k baseline samples from test subject
        baseline_mask_test = (y_test == 0)
        baseline_idx = np.where(baseline_mask_test)[0]
        norm_idx = rng.choice(baseline_idx, k_baseline, replace=False)
        eval_idx = np.setdiff1d(np.arange(len(y_test)), norm_idx)

        baseline = X_test[norm_idx].mean(axis=0)
        X_test_norm = X_test[eval_idx] - baseline
        y_test_eval = y_test[eval_idx]


        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_norm)
        X_test_scaled = scaler.transform(X_test_norm)


        # inner CV for hyperparameter tuning
        # define the grid search with inner CV
        search = GridSearchCV(
            estimator = model,
            param_grid = space,
            cv = inner_cv.split(X_train_scaled, y_train, g_train),
            scoring = 'accuracy' if model_type == 'classifier'
                else 'neg_mean_squared_error',
            refit=True,
            n_jobs=-1,
        )

        # execute the grid search
        result = search.fit(X_train_scaled, y_train)

        # get the best model from the grid search
        best_model = result.best_estimator_

        # evaluate the best model on the test set
        if model_type == 'classifier':
            y_pred = best_model.predict(X_test_scaled)
            y_score = best_model.predict_proba(X_test_scaled)[:, 1]
        if model_type == 'regressor':
            y_score = best_model.predict(X_test_scaled)
            y_pred = np.clip(np.round(y_score), 0, 1).astype(int)

        # AUC needs both classes to be present
        if len(np.unique(y_test_eval)) < 2:
            continue

        #store the results
        fold_acc.append(accuracy_score(y_test_eval, y_pred))
        fold_f1.append(f1_score(y_test_eval, y_pred, zero_division=0))
        fold_auc.append(roc_auc_score(y_test_eval, y_score))

        # sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_test_eval, y_pred, labels=[0, 1]).ravel()
        fold_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fold_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

        # store per-subject results for statistical testing
        fold_records.append({
            'subject': test_subject,
            'k_baseline': k_baseline,
            'accuracy': fold_acc[-1],
            'f1': fold_f1[-1],
            'auc': fold_auc[-1],
            'sensitivity': fold_sens[-1],
            'specificity': fold_spec[-1],
        })

        y_true_all.append(y_test_eval)
        y_pred_all.append(y_pred)
        y_score_all.append(y_score)

    # save the per-fold results
    save_fold_results(fold_records, csv_path_folds)

    # return the results
    return {
        'accuracy': np.mean(fold_acc),
        'f1': np.mean(fold_f1),
        'auc': np.mean(fold_auc),
        'sensitivity': np.mean(fold_sens),
        'specificity': np.mean(fold_spec),
        'accuracy_std': np.std(fold_acc),
        'f1_std': np.std(fold_f1),
        'auc_std': np.std(fold_auc),
        'sensitivity_std': np.std(fold_sens),
        'specificity_std': np.std(fold_spec),
        'y_true': np.concatenate(y_true_all),
        'y_pred': np.concatenate(y_pred_all),
        'y_score': np.concatenate(y_score_all),
        'fold_results': pd.DataFrame(fold_records),
    }


def loso_multiclass_nested_cv(X, y, groups, model, space, classes, model_type='classifier', csv_path_folds=None):
    """
    Loso for multiclass classification
    """
    outer_cv = LeaveOneGroupOut()
    inner_cv = GroupKFold(n_splits=INNER_NSPLITS)

    fold_acc, fold_f1, fold_auc = [], [], []
    fold_mae, fold_rmse = [], []
    fold_records = []

    n_classes = len(classes)

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = groups[train_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # inner CV for hyperparameter tuning
        #define the grid search with inner CV
        search = GridSearchCV(
            estimator = model,
            param_grid = space,
            cv = inner_cv.split(X_train_scaled, y_train, g_train),
            scoring = 'accuracy' if model_type == 'classifier'
                else 'neg_mean_squared_error',
            refit=True,
            n_jobs=-1,
        )
        # execute the grid search
        result = search.fit(X_train_scaled, y_train)
        # get the best model from the grid search
        best_model = result.best_estimator_

        # evaluate the best model on the test set
        if model_type == 'classifier':
            y_pred = best_model.predict(X_test_scaled)
            y_prob = best_model.predict_proba(X_test_scaled)
            y_mae = np.mean(np.abs(y_test - y_pred))
            y_rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        if model_type == 'regressor':
            y_num = best_model.predict(X_test_scaled)
            y_pred = np.clip(np.round(y_num), 0, n_classes-1).astype(int)
            y_mae = np.mean(np.abs(y_test - y_num))
            y_rmse = np.sqrt(np.mean((y_test - y_num)**2))
            # calculate probability from distanzes to classes
            class_values = np.arange(n_classes)
            distances = np.abs(y_num.reshape(-1, 1) - class_values.reshape(1, -1))
            y_prob = softmax(-distances, axis=1)

        # AUC
        y_prob = np.asarray(y_prob)
        present_classes = np.unique(y_test)

        if len(present_classes) < 2:
            continue

        if len(present_classes) < n_classes:
                auc = roc_auc_score(
                    y_test, y_prob[:, present_classes],
                    multi_class='ovr', average='macro', labels=present_classes,
                )
        else:
            auc = roc_auc_score(
            y_test, y_prob, multi_class='ovr', average='macro',
            )

        # store the results
        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_f1.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        fold_auc.append(auc)
        fold_mae.append(y_mae)
        fold_rmse.append(y_rmse)

        # store per-subject results for statistical testing
        fold_records.append({
            'subject': groups[test_idx][0],
            'accuracy': fold_acc[-1],
            'f1': fold_f1[-1],
            'auc': fold_auc[-1],
            'mae': fold_mae[-1],
            'rmse': fold_rmse[-1],
        })

    # save the per-fold results
    save_fold_results(fold_records, csv_path_folds)

    # return the results
    return {
        'accuracy': np.mean(fold_acc),
        'f1': np.mean(fold_f1),
        'auc': np.mean(fold_auc),
        'mae': np.mean(fold_mae),
        'rmse': np.mean(fold_rmse),
        'accuracy_std': np.std(fold_acc),
        'f1_std': np.std(fold_f1),
        'auc_std': np.std(fold_auc),
        'mae_std': np.std(fold_mae),
        'rmse_std': np.std(fold_rmse),
        'fold_results': pd.DataFrame(fold_records),
    }



def loso_binary_calibrated_nested_cv(X, y, groups, model, space, k, balance=False, csv_path_folds=None):
    """
    Loso for calibration and binary classification.
    """

    outer_cv = LeaveOneGroupOut()
    inner_cv = GroupKFold(n_splits=INNER_NSPLITS)

    fold_acc, fold_f1, fold_auc = [], [], []
    fold_sens, fold_spec = [], []
    fold_records = []

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = groups[train_idx]
        g_test = groups[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # get calibrated train/test split
        X_train_calib, y_train_calib, g_train_calib, X_test_calib, y_test_calib = calibrated_extended_features(
            X_train_scaled, y_train, g_train,
            X_test_scaled, y_test, g_test,
            k
        )

        # inner CV for hyperparameter tuning
        # define the grid search with inner CV
        search = GridSearchCV(
            estimator=model,
            param_grid=space,
            cv=inner_cv.split(X_train_calib, y_train_calib, g_train_calib),
            scoring='accuracy',
            refit=True,
            n_jobs=-1,
        )

        # execute the grid search
        # if balance is True, give more weight to calibration sample
        if balance:
            n_train_original = len(y_train)
            n_calibration = len(y_train_calib) - n_train_original

            sample_weights = np.ones_like(y_train_calib, dtype=float)
            sample_weights[n_train_original:] = n_train_original / n_calibration

            result = search.fit(
                X_train_calib,
                y_train_calib,
                sample_weight=sample_weights
            )
        else:
            result = search.fit(
                X_train_calib,
                y_train_calib
            )

        # get the best model from the grid search
        best_model = result.best_estimator_

        # evaluate the best model on the test set
        y_pred = best_model.predict(X_test_calib)
        y_score = best_model.predict_proba(X_test_calib)[:, 1]

        # AUC needs both classes to be present
        if len(np.unique(y_test_calib)) < 2:
            continue

        # store the results
        fold_acc.append(accuracy_score(y_test_calib, y_pred))
        fold_f1.append(f1_score(y_test_calib, y_pred, zero_division=0))
        fold_auc.append(roc_auc_score(y_test_calib, y_score))

        # sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_test_calib, y_pred, labels=[0, 1]).ravel()
        fold_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fold_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

        # store per-subject results for statistical testing
        fold_records.append({
            'subject': g_test[0],
            'k': k,
            'accuracy': fold_acc[-1],
            'f1': fold_f1[-1],
            'auc': fold_auc[-1],
            'sensitivity': fold_sens[-1],
            'specificity': fold_spec[-1],
        })

    # save the per-fold results
    save_fold_results(fold_records, csv_path_folds)

    # return the results
    return {
        'accuracy': np.mean(fold_acc),
        'f1': np.mean(fold_f1),
        'auc': np.mean(fold_auc),
        'sensitivity': np.mean(fold_sens),
        'specificity': np.mean(fold_spec),
        'accuracy_std': np.std(fold_acc),
        'f1_std': np.std(fold_f1),
        'auc_std': np.std(fold_auc),
        'sensitivity_std': np.std(fold_sens),
        'specificity_std': np.std(fold_spec),
        'k': k,
        'fold_results': pd.DataFrame(fold_records),
    }


def calibrated_extended_features(X_train, y_train, g_train, X_test, y_test, g_test, k):
    """
    Add k samples from subject balanced acros classes in the training set
    """

    labels = np.unique(y_train)
    samples_per_class = k // len(labels)

    candidate_indices = []

    for label in labels:
        label_indices = np.where(y_test == label)[0]

        selected_indices = np.random.choice(
            label_indices,
            size=samples_per_class,
            replace=False
        )

        candidate_indices.extend(selected_indices)

    candidate_indices = np.array(candidate_indices)

    test_indices = np.setdiff1d(np.arange(len(y_test)), candidate_indices)

    X_train_calib = np.concatenate([X_train, X_test[candidate_indices]], axis=0)
    y_train_calib = np.concatenate([y_train, y_test[candidate_indices]], axis=0)
    g_train_calib = np.concatenate([g_train, g_test[candidate_indices]], axis=0)

    X_test_calib = X_test[test_indices]
    y_test_calib = y_test[test_indices]

    return X_train_calib, y_train_calib, g_train_calib, X_test_calib, y_test_calib



def loso_binary_fully_personalized_nested_cv(X, y, groups, model, space, model_type, csv_path_folds=None):
    """
    Loso for fully personalized binary classification.
    """

    outer_cv = StratifiedKFold(
    n_splits=INNER_NSPLITS,
    shuffle=True,
    random_state=42
)

    inner_cv = StratifiedKFold(
        n_splits=INNER_NSPLITS,
        shuffle=True,
        random_state=42
    )

    all_acc, all_f1, all_auc = [], [], []
    all_sens, all_spec = [], []
    subject_results = []
    fold_records = []

    for subject in np.unique(groups):
        mask = groups == subject
        X_subject = X[mask]
        y_subject = y[mask]

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_subject, y_subject)):
            X_train, X_test = X_subject[train_idx], X_subject[test_idx]
            y_train, y_test = y_subject[train_idx], y_subject[test_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Grid search for hyperparameter tuning
            search = GridSearchCV(
                estimator=model,
                param_grid=space,
                cv=inner_cv,
                scoring='accuracy' if model_type == 'classifier' else 'neg_mean_squared_error',
                refit=True,
                n_jobs=-1,
            )

            result = search.fit(X_train_scaled, y_train)
            best_model = result.best_estimator_

            # Evaluate on the test set
            if model_type == 'classifier':
                y_pred = best_model.predict(X_test_scaled)
                y_score = best_model.predict_proba(X_test_scaled)[:, 1]
            if model_type == 'regressor':
                y_score = best_model.predict(X_test_scaled)
                y_pred = np.clip(np.round(y_score), 0, 1).astype(int)

            # AUC needs both classes to be present
            if len(np.unique(y_test)) < 2:
                continue

            all_acc.append(accuracy_score(y_test, y_pred))
            all_f1.append(f1_score(y_test, y_pred, zero_division=0))
            all_auc.append(roc_auc_score(y_test, y_score))

            # sensitivity and specificity
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            all_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            all_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

            # store per-subject/per-fold results for statistical testing
            fold_records.append({
                'subject': subject,
                'fold': fold_idx,
                'accuracy': all_acc[-1],
                'f1': all_f1[-1],
                'auc': all_auc[-1],
                'sensitivity': all_sens[-1],
                'specificity': all_spec[-1],
            })

    # save the per-fold results
    save_fold_results(fold_records, csv_path_folds)

    all_results = {
        'accuracy': np.mean(all_acc),
        'f1': np.mean(all_f1),
        'auc': np.mean(all_auc),
        'sensitivity': np.mean(all_sens),
        'specificity': np.mean(all_spec),
        'accuracy_std': np.std(all_acc),
        'f1_std': np.std(all_f1),
        'auc_std': np.std(all_auc),
        'sensitivity_std': np.std(all_sens),
        'specificity_std': np.std(all_spec),
        'fold_results': pd.DataFrame(fold_records),
    }

    return all_results


def loso_binary_nested_cv_with_group_importance(X, y, groups, model, space, model_type, signal_groups, scoring, n_repeats=10, random_state=RANDOM_STATE, csv_path=None):
    """
    Loso for binary classification with group permutation importance per outer fold.
    """

    outer_cv = LeaveOneGroupOut()
    inner_cv = GroupKFold(n_splits=INNER_NSPLITS)

    fold_acc, fold_f1, fold_auc = [], [], []
    group_importance_records = []

    # scorer for permutation importance
    if scoring:
        permutation_scoring = scoring
    else:
        permutation_scoring = 'accuracy'
    scorer = get_scorer(permutation_scoring)

    fold_i = 0
    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = groups[train_idx]

        test_group = groups[test_idx][0]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # inner CV for hyperparameter tuning
        search = GridSearchCV(
            estimator=model,
            param_grid=space,
            cv=inner_cv.split(X_train_scaled, y_train, g_train),
            scoring='accuracy' if model_type == 'classifier' else 'neg_mean_squared_error',
            refit=True,
            n_jobs=-1,
        )

        # execute the grid search
        result = search.fit(X_train_scaled, y_train)

        # get the best model from the grid search
        best_model = result.best_estimator_

        # evaluate the best model on the test set
        if model_type == 'classifier':
            y_pred = best_model.predict(X_test_scaled)
            y_score = best_model.predict_proba(X_test_scaled)[:, 1]
        if model_type == 'regressor':
            y_score = best_model.predict(X_test_scaled)
            y_pred = np.clip(np.round(y_score), 0, 1).astype(int)

        # store accuracy and f1
        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_f1.append(f1_score(y_test, y_pred, zero_division=0))

        # AUC needs both classes to be present
        fold_auc_value = np.nan
        if len(np.unique(y_test)) >= 2:
            fold_auc_value = roc_auc_score(y_test, y_score)
            fold_auc.append(fold_auc_value)


        # group permutation importance
        base_score = scorer(best_model, X_test_scaled, y_test)

        rng = np.random.RandomState(random_state + fold_i)

        for signal_name, idxs in signal_groups.items():
            idxs = np.asarray(idxs, dtype=int)
            drops = []

            for _ in range(n_repeats):
                X_permutation = X_test_scaled.copy()
                permutation = rng.permutation(len(X_permutation))

                # permute all features of the signal group together
                X_permutation[:, idxs] = X_test_scaled[permutation][:, idxs]

                perm_score = scorer(best_model, X_permutation, y_test)
                drops.append(base_score - perm_score)

            group_importance_records.append({
                'fold': fold_i,
                'test_group': test_group,
                'signal': signal_name,
                'n_features': len(idxs),
                'importance_mean': float(np.mean(drops)),
                'importance_std': float(np.std(drops)),
                'base_score': float(base_score),
                'scoring': permutation_scoring,
            })

        fold_i += 1


    group_importance_df = pd.DataFrame(group_importance_records)

    group_importance = (
        group_importance_df
        .groupby('signal', as_index=False)
        .agg(
            importance_mean=('importance_mean', 'mean'),
            importance_std=('importance_mean', 'std'),
            n_folds=('importance_mean', 'count'),
            n_features=('n_features', 'first'),
        )
        .sort_values('importance_mean', ascending=False)
    )

    return {
        'accuracy': np.mean(fold_acc),
        'f1': np.mean(fold_f1),
        'auc': np.mean(fold_auc),
        'accuracy_std': np.std(fold_acc),
        'f1_std': np.std(fold_f1),
        'auc_std': np.std(fold_auc),
        'group_importance': group_importance,
    }
