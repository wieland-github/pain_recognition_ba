import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.special import softmax

#-----------------------------------
# Define constants
#-----------------------------------
INNER_NSPLITS = 5
RANDOM_STATE = 42



# -----------------------------------
# Define functions for model training
# -----------------------------------

def loso_binary_nested_cv(X, y, groups, model, space, model_type):
    """
    Loso for binary classification
    """

    outer_cv = LeaveOneGroupOut()
    inner_cv = GroupKFold(n_splits=INNER_NSPLITS)

    fold_acc, fold_f1, fold_auc = [], [], []

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

    # return the results 
    return {
        'accuracy': np.mean(fold_acc),            
        'f1': np.mean(fold_f1),
        'auc': np.mean(fold_auc),
        'accuracy_std': np.std(fold_acc),
        'f1_std': np.std(fold_f1),
        'auc_std': np.std(fold_auc)
    }


def loso_multiclass_nested_cv(X, y, groups, model, space, classes, model_type='classifier'):
    """
    Loso for multiclass classification
    """
    outer_cv = LeaveOneGroupOut()
    inner_cv = GroupKFold(n_splits=INNER_NSPLITS)

    fold_acc, fold_f1, fold_auc = [], [], []
    fold_mae, fold_rmse = [], []

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
                    multi_class="ovr", average="macro", labels=present_classes,
                )
        else:
            auc = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="macro",
            )

        # store the results
        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_f1.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        fold_auc.append(auc)
        fold_mae.append(y_mae)
        fold_rmse.append(y_rmse)

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
    } 



def loso_binary_calibrated_nested_cv(X, y, groups, model, space, k):
    """
    Loso for calibration and binary classification
    """

    outer_cv = LeaveOneGroupOut()
    inner_cv = GroupKFold(n_splits=INNER_NSPLITS)

    fold_acc, fold_f1, fold_auc = [], [], []

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
        )

        # execute the grid search
        result = search.fit(X_train_calib, y_train_calib)

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

    # return the results
    return {
        'accuracy': np.mean(fold_acc),
        'f1': np.mean(fold_f1),
        'auc': np.mean(fold_auc),
        'accuracy_std': np.std(fold_acc),
        'f1_std': np.std(fold_f1),
        'auc_std': np.std(fold_auc),
        'k': k
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



def safe_results_binary(model_name, personalization, metrics):
    """
    safe the reults for later use
    """

    results = {
        'Model': model_name,
        'Personalization': personalization,
        'K': metrics.get('k', None),
        'Accuracy': metrics.get('accuracy'),
        'Accuracy_Std': metrics.get('accuracy_std'),
        'F1': metrics.get('f1'),
        'F1_Std': metrics.get('f1_std'),
        'AUC': metrics.get('auc'),
        'AUC_Std': metrics.get('auc_std')
    }
    
    # filename
    filename = f"test_{model_name}_{personalization}.csv"

    # safe
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

    return results