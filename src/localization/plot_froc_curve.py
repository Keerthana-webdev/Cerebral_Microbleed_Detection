"""
Lesson 14e: Plot the sensitivity vs false-positives-per-subject tradeoff
as a FROC-style curve — the standard way your literature survey's papers
report this exact tradeoff. Turns raw numbers into a defensible chart.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
df = pd.read_csv(REPORTS_DIR / "threshold_sweep.csv")

# For each FP level (roughly), keep only the best sensitivity achieved
df_sorted = df.sort_values("fp_per_subject")

plt.figure(figsize=(8, 6))
plt.scatter(df["fp_per_subject"], df["sensitivity"], alpha=0.6, s=40)
plt.plot(df_sorted["fp_per_subject"], df_sorted["sensitivity"].cummax(), color="red", linewidth=2, label="Best achievable frontier")
plt.xlabel("False Positives per Subject")
plt.ylabel("Sensitivity")
plt.title("Whole-Scan Localization: Sensitivity vs False-Positive Tradeoff (FROC-style)")
plt.legend()
plt.grid(alpha=0.3)

out_path = REPORTS_DIR / "froc_curve.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved FROC curve to: {out_path}")
plt.show()