"""
Lesson 12a: Grade each test subject's severity based on how many CMBs our
full pipeline detected. Simple, rule-based, and clinically interpretable —
no black-box model involved in this step, on purpose.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PATCHES_DIR


def grade_severity(cmb_count):
    """
    Simple count-based severity bands, inspired by clinical CMB rating
    approaches (e.g., Greenberg et al.). Thresholds are a reasonable
    starting point for a capstone project — document this choice in your
    report rather than presenting it as a validated clinical scale.
    """
    if cmb_count == 0:
        return "None"
    elif cmb_count <= 2:
        return "Mild"
    elif cmb_count <= 5:
        return "Moderate"
    else:
        return "Severe"


def main():
    results_path = PATCHES_DIR / "confidence_evaluation_results.csv"
    df = pd.read_csv(results_path)

    # Count how many patches the FULL PIPELINE predicted as microbleeds, per subject
    predicted_counts = df[df["predicted_label"] == 1].groupby("subject").size()
    # Also count the TRUE number, so we can sanity-check our grading against ground truth
    true_counts = df[df["true_label"] == 1].groupby("subject").size()

    all_subjects = sorted(df["subject"].unique())
    rows = []
    for subject in all_subjects:
        predicted = int(predicted_counts.get(subject, 0))
        true = int(true_counts.get(subject, 0))
        rows.append({
            "subject": subject,
            "predicted_cmb_count": predicted,
            "true_cmb_count": true,
            "predicted_severity": grade_severity(predicted),
            "true_severity": grade_severity(true),
        })

    severity_df = pd.DataFrame(rows)
    print(severity_df.to_string(index=False))

    agreement = (severity_df["predicted_severity"] == severity_df["true_severity"]).mean()
    print(f"\nSeverity grade agreement with ground truth: {agreement:.1%}")

    out_path = PATCHES_DIR / "severity_report.csv"
    severity_df.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()