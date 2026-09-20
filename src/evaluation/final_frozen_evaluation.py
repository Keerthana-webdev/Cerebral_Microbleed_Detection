"""
THE FINAL EVALUATION. Run once, on the untouched test set, using the
final v3 model (deeper CNN + round 1+2 hard negatives) + final chosen
thresholds. No further tuning after this. Reports every metric needed
for the final report.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from config import PROCESSED_DATA_DIR, PATCHES_DIR, SPLITS_DIR
from model_v3 import DeeperCNN3D
from model import SimpleCNN3D

DEVICE = torch.device("cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)
STAGE1_THRESHOLD = 0.8
STAGE2_THRESHOLD = 0.8
NMS_DISTANCE = 25
MATCH_DISTANCE = 6

stage1_model = DeeperCNN3D().to(DEVICE)
stage1_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_v3_best.pt", map_location=DEVICE))
stage1_model.eval()

stage2_model = SimpleCNN3D().to(DEVICE)
stage2_model.load_state_dict(torch.load(MODELS_DIR / "mimic_classifier_best.pt", map_location=DEVICE))
stage2_model.eval()


def cut_patch(volume, center, size):
    half = [s // 2 for s in size]
    slices = tuple(slice(c - h, c + h) for c, h in zip(center, half))
    return volume[slices]


def generate_grid(shape):
    half = [s // 2 for s in PATCH_SIZE]
    centers = []
    for x in range(half[0], shape[0] - half[0], STRIDE[0]):
        for y in range(half[1], shape[1] - half[1], STRIDE[1]):
            for z in range(half[2], shape[2] - half[2], STRIDE[2]):
                centers.append((x, y, z))
    return centers


def get_true_centers(mask):
    labeled, n = ndimage.label(mask)
    return [tuple(np.argwhere(labeled == i).mean(axis=0)) for i in range(1, n + 1) if (labeled == i).sum() >= 1]


def apply_nms(centers, probs, min_distance):
    order = np.argsort(-np.array(probs))
    kept = []
    for i in order:
        c = centers[i]
        if not any(np.linalg.norm(np.array(c) - np.array(k)) < min_distance for k in kept):
            kept.append(c)
    return kept


def match(predicted, true, max_dist):
    matched = set()
    tp = 0
    for p in predicted:
        for i, t in enumerate(true):
            if i in matched:
                continue
            if np.linalg.norm(np.array(p) - np.array(t)) <= max_dist:
                tp += 1
                matched.add(i)
                break
    fp = len(predicted) - tp
    fn = len(true) - len(matched)
    return tp, fp, fn


def patch_level_evaluation(test_subjects):
    metadata = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
    test_metadata = metadata[metadata["subject"].isin(test_subjects)].reset_index(drop=True)

    tp = fp = tn = fn = 0
    with torch.no_grad():
        for _, row in test_metadata.iterrows():
            patch = np.load(PATCHES_DIR / row["subject"] / row["file"])
            tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            true_label = row["label"]

            s1 = torch.sigmoid(stage1_model(tensor)).item()
            if s1 > STAGE1_THRESHOLD:
                s2 = torch.sigmoid(stage2_model(tensor)).item()
                predicted = 1 if s2 > STAGE2_THRESHOLD else 0
            else:
                predicted = 0

            if predicted == 1 and true_label == 1: tp += 1
            elif predicted == 1 and true_label == 0: fp += 1
            elif predicted == 0 and true_label == 0: tn += 1
            elif predicted == 0 and true_label == 1: fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("=" * 70)
    print("PART A: PATCH-LEVEL METRICS (curated test patches, final v3+mimic pipeline)")
    print("=" * 70)
    print(f"Confusion Matrix:")
    print(f"                  Predicted CMB   Predicted Normal")
    print(f"  Actual CMB           {tp:4d}              {fn:4d}")
    print(f"  Actual Normal        {fp:4d}              {tn:4d}")
    print(f"\nPrecision:    {precision:.3f}")
    print(f"Recall/Sens:  {recall:.3f}")
    print(f"Specificity:  {specificity:.3f}")
    print(f"Accuracy:     {accuracy:.3f}")
    print(f"F1-Score:     {f1:.3f}\n")

    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=round(precision, 3),
                recall=round(recall, 3), specificity=round(specificity, 3),
                accuracy=round(accuracy, 3), f1_score=round(f1, 3))


def wholescan_evaluation(test_subjects):
    print("=" * 70)
    print("PART B: WHOLE-SCAN LEVEL METRICS (final v3 pipeline, per-subject)")
    print("=" * 70)

    per_subject_rows = []
    total_tp = total_fp = total_fn = 0

    for subject_name in test_subjects:
        volume = np.load(PROCESSED_DATA_DIR / subject_name / "swi.npy")
        mask = np.load(PROCESSED_DATA_DIR / subject_name / "mask.npy")
        true_centers = get_true_centers(mask)

        all_centers = generate_grid(volume.shape)
        candidates = [c for c in all_centers if cut_patch(volume, c, PATCH_SIZE).std() > 0.05]

        stage1_hits = []
        for i in range(0, len(candidates), 64):
            batch = candidates[i:i + 64]
            patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in batch])
            tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                probs = torch.sigmoid(stage1_model(tensor)).squeeze(1).numpy()
            stage1_hits.extend([(c, p) for c, p in zip(batch, probs) if p > STAGE1_THRESHOLD])

        final = []
        for i in range(0, len(stage1_hits), 64):
            batch = stage1_hits[i:i + 64]
            batch_centers = [c for c, p in batch]
            patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in batch_centers])
            tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                probs = torch.sigmoid(stage2_model(tensor)).squeeze(1).numpy()
            for c, p in zip(batch_centers, probs):
                if p > STAGE2_THRESHOLD:
                    final.append((c, float(p)))

        if final:
            centers, probs = zip(*final)
            kept = apply_nms(list(centers), list(probs), NMS_DISTANCE)
        else:
            kept = []

        tp, fp, fn = match(kept, true_centers, MATCH_DISTANCE)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None

        per_subject_rows.append({
            "subject": subject_name, "true_lesions": len(true_centers),
            "detections": len(kept), "tp": tp, "fp": fp, "fn": fn,
            "sensitivity": round(sensitivity, 3) if sensitivity is not None else "N/A",
        })
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(f"  {subject_name}: true={len(true_centers)}, detected={len(kept)}, "
              f"TP={tp}, FP={fp}, FN={fn}")

    overall_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    fp_per_scan = total_fp / len(test_subjects)
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    print(f"\nOVERALL WHOLE-SCAN RESULTS:")
    print(f"  Sensitivity (lesion-level):  {overall_sensitivity:.3f}")
    print(f"  Precision (lesion-level):    {overall_precision:.3f}")
    print(f"  False Positives per Scan:    {fp_per_scan:.2f}")
    print(f"  Total TP={total_tp}, FP={total_fp}, FN={total_fn}\n")

    per_subject_df = pd.DataFrame(per_subject_rows)
    per_subject_df.to_csv(REPORTS_DIR / "final_per_subject_results.csv", index=False)
    print(f"Per-subject results saved to: {REPORTS_DIR / 'final_per_subject_results.csv'}")

    return {
        "sensitivity": round(overall_sensitivity, 3),
        "precision": round(overall_precision, 3),
        "fp_per_scan": round(fp_per_scan, 2),
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
    }, per_subject_df


def main():
    with open(SPLITS_DIR / "splits.json") as f:
        splits = json.load(f)
    test_subjects = splits["test"]

    print(f"\n🔒 FINAL FROZEN EVALUATION (v3) — test set ({len(test_subjects)} subjects), no further tuning after this.\n")

    patch_metrics = patch_level_evaluation(test_subjects)
    scan_metrics, per_subject_df = wholescan_evaluation(test_subjects)

    summary = {
        "model_version": "v3",
        "operating_point": {
            "stage1_threshold": STAGE1_THRESHOLD,
            "stage2_threshold": STAGE2_THRESHOLD,
            "nms_distance": NMS_DISTANCE,
        },
        "patch_level": patch_metrics,
        "whole_scan_level": scan_metrics,
    }

    out_path = REPORTS_DIR / "FINAL_RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"✅ FINAL RESULTS SAVED TO: {out_path}")
    print("These are the numbers for your final report. Do not re-tune after this.")
    print("=" * 70)


if __name__ == "__main__":
    main()