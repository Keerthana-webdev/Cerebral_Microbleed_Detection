"""
Lesson 4: Clean up EVERY subject's scan (not just sub-101) and save the
cleaned versions as .npy files (a fast, simple numpy array format) into
data/preprocessed/ — ready for AI training later.
"""

import numpy as np
import SimpleITK as sitk
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SWI_PATTERN, MASK_PATTERN, TARGET_SPACING, INTENSITY_CLIP_PERCENTILES





def resize_to_standard_spacing(image, is_mask=False):
    """Resampling: makes every scan use the same real-world voxel size."""
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [
        int(round(size * old / new))
        for size, old, new in zip(old_size, old_spacing, TARGET_SPACING)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    # Masks use nearest-neighbor (keeps 0/1 values crisp); scans use smooth interpolation
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resampler.Execute(image)


def normalize_brightness(array):
    """Normalization: rescales brightness to a standard range across all patients."""
    low, high = np.percentile(array, INTENSITY_CLIP_PERCENTILES)
    array = np.clip(array, low, high)          # remove extreme outlier brightness spikes
    mean, std = array.mean(), array.std()
    return (array - mean) / (std + 1e-8)         # now centered at 0, standard spread of 1


def process_one_subject(subject_name):
    subject_folder = RAW_DATA_DIR / subject_name
    swi_path = subject_folder / SWI_PATTERN.format(subj=subject_name)
    mask_path = subject_folder / MASK_PATTERN.format(subj=subject_name)

    swi_image = sitk.ReadImage(str(swi_path))
    swi_image = fix_uneven_brightness(swi_image)
    swi_image = resize_to_standard_spacing(swi_image, is_mask=False)

    if mask_path.exists():
        mask_image = sitk.ReadImage(str(mask_path))
        mask_image = resize_to_standard_spacing(mask_image, is_mask=True)
        mask_array = sitk.GetArrayFromImage(mask_image)
    else:
        # CHANGED: no mask file = this patient has zero microbleeds.
        # Build an all-zero mask with the same shape as the (already resized) scan.
        print(f"    ⚠️  No CMB mask for {subject_name} — treating as zero-microbleed subject")
        swi_array_shape = sitk.GetArrayFromImage(swi_image).shape
        mask_array = np.zeros(swi_array_shape, dtype=np.uint8)

    swi_array = sitk.GetArrayFromImage(swi_image)
    swi_array = normalize_brightness(swi_array)

    out_folder = PROCESSED_DATA_DIR / subject_name
    out_folder.mkdir(parents=True, exist_ok=True)
    np.save(out_folder / "swi.npy", swi_array.astype(np.float32))
    np.save(out_folder / "mask.npy", mask_array.astype(np.uint8))


def main():
    # Find every real subject folder (skip junk files like ._sub-201)
    subjects = sorted([
        d.name for d in RAW_DATA_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("._")
    ])
    print(f"Found {len(subjects)} subjects. Processing each one...")

    succeeded, failed = 0, []
    for i, subject_name in enumerate(subjects, start=1):
        try:
            process_one_subject(subject_name)
            succeeded += 1
            print(f"[{i}/{len(subjects)}] ✅ {subject_name} done")
        except Exception as e:
            failed.append((subject_name, str(e)))
            print(f"[{i}/{len(subjects)}] ❌ {subject_name} FAILED: {e}")

    print(f"\nDone. {succeeded}/{len(subjects)} succeeded.")
    if failed:
        print("Failed subjects:", failed)


if __name__ == "__main__":
    main()