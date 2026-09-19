"""
Lesson 5: Split all 72 subjects into train/val/test groups.
Splitting by SUBJECT (not by slice) so no patient's data leaks across sets.
"""

import json
import random
from config import PROCESSED_DATA_DIR, SPLITS_DIR

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# whatever's left over automatically becomes TEST
RANDOM_SEED = 42   # fixed number so the split is the same every time we run this


def main():
    # Get every subject we successfully preprocessed
    subjects = sorted([
        d.name for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()
    ])
    print(f"Total preprocessed subjects: {len(subjects)}")

    random.seed(RANDOM_SEED)
    random.shuffle(subjects)

    n_total = len(subjects)
    n_train = round(n_total * TRAIN_FRACTION)
    n_val = round(n_total * VAL_FRACTION)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    splits = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SPLITS_DIR / "splits.json"
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Train: {len(train_subjects)} subjects")
    print(f"Val:   {len(val_subjects)} subjects")
    print(f"Test:  {len(test_subjects)} subjects")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()