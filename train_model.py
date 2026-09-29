"""
Smart Attendance AI - Model Training (Enhanced)
=================================================
Enhancements over the original version:
- Five candidate models instead of two (Logistic Regression, Random Forest,
  Gradient Boosting, SVM, KNN)
- Stratified 5-fold cross-validation for a more reliable comparison than a
  single train/test split
- Light hyperparameter tuning (GridSearchCV) on the two strongest baseline
  models before final selection
- Feature-importance chart for the winning model (when supported)
- A model-comparison bar chart, saved alongside the existing CSV
- All metrics + the chosen model's parameters saved to model/metrics.json,
  so app.py (or anyone else) can display "why this model" without re-running
  training
"""

import json
import logging
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("train_model")

DATA_PATH = "dataset/cleaned_attendance_data.csv"
MODEL_DIR = "model"
CV_FOLDS = 5
RANDOM_STATE = 42


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data["Overall_Attendance"] = (
        data["Math_Attendance"] + data["Python_Attendance"] + data["DBMS_Attendance"] + data["AI_Attendance"]
    ) / 4
    data["Attendance_Change"] = data["Attendance_Percentage"] - data["Previous_Attendance"]
    data["Academic_Performance"] = (data["Internal_Marks"] + data["Assignment_Percentage"]) / 2
    return data


def evaluate(model_name, y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    log.info("=== %s ===", model_name)
    for k, v in metrics.items():
        log.info("%s: %.4f", k, v)
    log.info("\n%s", classification_report(y_true, y_pred, zero_division=0))
    log.info("Confusion matrix:\n%s", confusion_matrix(y_true, y_pred))
    return metrics


def main():
    log.info("Loading cleaned dataset from %s", DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    data = engineer_features(data)

    features = [
        "Attendance_Percentage",
        "Math_Attendance",
        "Python_Attendance",
        "DBMS_Attendance",
        "AI_Attendance",
        "Internal_Marks",
        "Assignment_Percentage",
        "Previous_Attendance",
        "Medical_Leave",
        "Overall_Attendance",
        "Attendance_Change",
        "Academic_Performance",
    ]
    X = data[features]
    y = data["Risk_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    log.info("Training records: %d | Testing records: %d", len(X_train), len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- Baseline candidates, compared with cross-validation ------------------
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }

    cv_results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1_weighted")
        cv_results[name] = {"mean_f1": float(scores.mean()), "std_f1": float(scores.std())}
        log.info("[CV] %-20s  F1 = %.4f (+/- %.4f)", name, scores.mean(), scores.std())

    # --- Light hyperparameter tuning on the two most promising model families -
    log.info("Tuning Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={"n_estimators": [150, 250], "max_depth": [None, 10, 20]},
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    rf_grid.fit(X_train_scaled, y_train)
    log.info("Best RF params: %s (CV F1=%.4f)", rf_grid.best_params_, rf_grid.best_score_)

    log.info("Tuning Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid={"C": [0.1, 1.0, 5.0]},
        cv=cv,
        scoring="f1_weighted",
    )
    lr_grid.fit(X_train_scaled, y_train)
    log.info("Best LR params: %s (CV F1=%.4f)", lr_grid.best_params_, lr_grid.best_score_)

    tuned_models = {
        "Random Forest (tuned)": rf_grid.best_estimator_,
        "Logistic Regression (tuned)": lr_grid.best_estimator_,
    }

    # --- Final evaluation on the held-out test set for every candidate --------
    all_models = {**candidates, **tuned_models}
    comparison_rows = []
    test_metrics_by_model = {}

    for name, model in all_models.items():
        if name not in tuned_models:
            model.fit(X_train_scaled, y_train)  # re-fit baseline on full train set
        prediction = model.predict(X_test_scaled)
        metrics = evaluate(name, y_test, prediction)
        test_metrics_by_model[name] = metrics
        comparison_rows.append({"Model": name, **{k.capitalize(): v for k, v in metrics.items()}})

    comparison = pd.DataFrame(comparison_rows).sort_values("F1", ascending=False).reset_index(drop=True)
    log.info("\n=== MODEL COMPARISON (test set) ===\n%s", comparison.to_string(index=False))

    best_model_name = comparison.iloc[0]["Model"]
    best_model = all_models[best_model_name]
    log.info("Selected best model: %s", best_model_name)

    # --- Persist everything ----------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "features.pkl"))
    comparison.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)

    metrics_payload = {
        "best_model": best_model_name,
        "cross_validation_f1_by_model": cv_results,
        "test_set_metrics_by_model": test_metrics_by_model,
        "best_model_params": {k: str(v) for k, v in best_model.get_params().items()},
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # --- Comparison chart --------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(comparison["Model"], comparison["F1"], color="#2563eb")
    plt.ylabel("Weighted F1 (test set)")
    plt.title("Model Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # --- Feature importance (only for tree-based winners) ----------------------
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=features).sort_values()
        plt.figure(figsize=(9, 6))
        importances.plot(kind="barh", color="#2563eb")
        plt.title(f"Feature Importance - {best_model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
        plt.close()
        log.info("Feature importance chart saved.")
    else:
        log.info("Best model has no native feature_importances_; skipping that chart.")

    log.info("MODEL TRAINING COMPLETED")
    log.info("Best model: %s -> model/best_model.pkl", best_model_name)
    log.info("Metrics: model/metrics.json | Comparison: model/model_comparison.csv/.png")


if __name__ == "__main__":
    main()