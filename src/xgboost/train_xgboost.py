import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "outputs" / "xgboost"
MODEL_DIR = PROJECT_ROOT / "models" / "xgboost"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_prob, threshold=0.5):

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )
    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(y_true, y_prob)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob):

    best_threshold = 0.5
    best_f1 = -1
    best_metrics = None

    thresholds = np.arange(
        0.10,
        0.91,
        0.01
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            y_prob,
            threshold
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def main():

    print("=" * 70)
    print("XGBOOST CMB CLASSIFIER")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load extracted features
    # ---------------------------------------------------------

    X_train = np.load(DATA_DIR / "train_X.npy")
    y_train = np.load(DATA_DIR / "train_y.npy")

    X_val = np.load(DATA_DIR / "val_X.npy")
    y_val = np.load(DATA_DIR / "val_y.npy")

    X_test = np.load(DATA_DIR / "test_X.npy")
    y_test = np.load(DATA_DIR / "test_y.npy")

    print("\nDataset:")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    print("\nClass distribution:")
    print("Train positives:", np.sum(y_train == 1))
    print("Train negatives:", np.sum(y_train == 0))

    # ---------------------------------------------------------
    # Handle class imbalance
    # ---------------------------------------------------------

    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)

    scale_pos_weight = negative_count / positive_count

    print(
        "\nscale_pos_weight:",
        round(scale_pos_weight, 4)
    )

    # ---------------------------------------------------------
    # Create XGBoost model
    # ---------------------------------------------------------

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,

        objective="binary:logistic",

        eval_metric="auc",

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",

        random_state=42,

        n_jobs=-1,
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("\nTraining XGBoost...")
    print("-" * 70)

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_train, y_train),
            (X_val, y_val),
        ],

        verbose=True,
    )

    print("\nTraining completed.")

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    val_prob = model.predict_proba(X_val)[:, 1]

    val_threshold, val_metrics = find_best_threshold(
        y_val,
        val_prob
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    for key, value in val_metrics.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # Final TEST evaluation
    # ---------------------------------------------------------

    test_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_prob,
        val_threshold
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Threshold   : {test_metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy    : {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : {test_metrics['precision']:.4f}"
    )

    print(
        f"Sensitivity : {test_metrics['sensitivity_recall']:.4f}"
    )

    print(
        f"Specificity : {test_metrics['specificity']:.4f}"
    )

    print(
        f"F1 Score    : {test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC     : {test_metrics['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        f"TN={test_metrics['tn']} "
        f"FP={test_metrics['fp']} "
        f"FN={test_metrics['fn']} "
        f"TP={test_metrics['tp']}"
    )

    model_path = MODEL_DIR / "xgboost_cmb.json"

    model.save_model(model_path)

    print("\nModel saved:")
    print(model_path)

    test_predictions = (
        test_prob >= val_threshold
    ).astype(int)

    prediction_file = DATA_DIR / "test_predictions.csv"

    np.savetxt(
        prediction_file,
        np.column_stack([
            y_test,
            test_prob,
            test_predictions
        ]),
        delimiter=",",
        header="true_label,probability,predicted_label",
        comments=""
    )

    print("Predictions saved:")
    print(prediction_file)

    report = {
        "model": "XGBoost",
        "feature_count": int(X_train.shape[1]),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    report_path = REPORT_DIR / "xgboost_results.json"

    with open(report_path, "w") as f:
        json.dump(
            report,
            f,
            indent=4
        )

    print("\nReport saved:")
    print(report_path)

    print("\n" + "=" * 70)
    print("XGBOOST EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()