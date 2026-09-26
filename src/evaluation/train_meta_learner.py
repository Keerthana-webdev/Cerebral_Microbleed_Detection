"""
Lesson 28b: Train a logistic regression meta-learner on VALIDATION
patch probabilities (CNN v3, ResNet, DenseNet), then apply it ONCE to
the already-computed TEST probabilities. This is the final ensemble
technique tested in this project.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---- Train on VALIDATION ----
val_df = pd.read_csv(REPORTS_DIR / "meta_learner_validation_features.csv")

X_val = val_df[["p_cnn", "p_resnet", "p_densenet"]].values
y_val = val_df["true_label"].values

meta_learner = LogisticRegression(class_weight="balanced", max_iter=1000)
meta_learner.fit(X_val, y_val)

print("=" * 70)
print("META-LEARNER TRAINED ON VALIDATION SET")
print("=" * 70)
print(f"Learned coefficients: CNN={meta_learner.coef_[0][0]:.3f}, "
      f"ResNet={meta_learner.coef_[0][1]:.3f}, DenseNet={meta_learner.coef_[0][2]:.3f}")
print(f"Intercept: {meta_learner.intercept_[0]:.3f}")

# ---- Apply ONCE to already-computed TEST probabilities ----
# (from your earlier ensemble_patch_evaluation.py run - reuse that CSV)
test_df = pd.read_csv(REPORTS_DIR / "ensemble_patch_predictions.csv")

X_test = test_df[["p_cnn", "p_resnet", "p_densenet"]].values
y_test = test_df["true_label"].values

predicted_probs = meta_learner.predict_proba(X_test)[:, 1]
predicted_labels = (predicted_probs > 0.5).astype(int)

tp = ((predicted_labels == 1) & (y_test == 1)).sum()
fp = ((predicted_labels == 1) & (y_test == 0)).sum()
tn = ((predicted_labels == 0) & (y_test == 0)).sum()
fn = ((predicted_labels == 0) & (y_test == 1)).sum()

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)
accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("\n" + "=" * 70)
print("META-LEARNER (STACKED) ENSEMBLE — FINAL TEST RESULTS")
print("=" * 70)
print(f"Confusion Matrix:")
print(f"                  Predicted CMB   Predicted Normal")
print(f"  Actual CMB           {tp:4d}              {fn:4d}")
print(f"  Actual Normal        {fp:4d}              {tn:4d}")
print(f"\nPrecision:    {precision:.4f}")
print(f"Recall/Sens:  {recall:.4f}")
print(f"Specificity:  {specificity:.4f}")
print(f"Accuracy:     {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"F1-Score:     {f1:.4f}")

print("\n=== FINAL COMPARISON — all techniques tried, same frozen test set ===")
print(f"3D CNN v3 alone:              Accuracy 98.20%, F1 93.60%")
print(f"3D ResNet alone:              Accuracy 96.65%, F1 89.92%")
print(f"3D DenseNet alone:            Accuracy 96.39%, F1 88.71%")
print(f"Equal-vote ensemble:          Accuracy 97.42%, F1 91.94%")
print(f"F1-weighted ensemble:         Accuracy 97.42%, F1 91.94%")
print(f"Meta-learner (stacked):       Accuracy {accuracy*100:.2f}%, F1 {f1*100:.2f}%")

test_df["meta_prob"] = predicted_probs
test_df["meta_predicted"] = predicted_labels
test_df.to_csv(REPORTS_DIR / "meta_learner_test_predictions.csv", index=False)
print(f"\nSaved to: {REPORTS_DIR / 'meta_learner_test_predictions.csv'}")