"""
Lesson 3: Plot a real slice of the brain scan, with the microbleed
locations highlighted in red — so we can SEE what we're working with.
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from config import RAW_DATA_DIR, SWI_PATTERN, MASK_PATTERN

subject_name = "sub-101"
subject_folder = RAW_DATA_DIR / subject_name

swi_image = nib.load(str(subject_folder / SWI_PATTERN.format(subj=subject_name)))
mask_image = nib.load(str(subject_folder / MASK_PATTERN.format(subj=subject_name)))

swi_data = swi_image.get_fdata()    # shape: (512, 512, 35)
mask_data = mask_image.get_fdata()  # same shape, 1 = microbleed, 0 = normal

# Find which slice (out of 35) actually contains a microbleed, so we look at
# a slice that has something interesting instead of a random empty one.
voxels_per_slice = mask_data.sum(axis=(0, 1))   # add up mask "1"s in each slice
slice_with_lesion = int(np.argmax(voxels_per_slice))  # index of the busiest slice
print("Showing slice number:", slice_with_lesion)
print("Microbleed voxels in this slice:", int(voxels_per_slice[slice_with_lesion]))

brain_slice = swi_data[:, :, slice_with_lesion]
mask_slice = mask_data[:, :, slice_with_lesion]

# Plot: brain scan in grayscale, microbleed pixels highlighted in red
plt.figure(figsize=(6, 6))
plt.imshow(brain_slice.T, cmap="gray", origin="lower")

# Overlay the mask in red wherever it's 1, transparent (nan) elsewhere
mask_overlay = np.where(mask_slice.T > 0, 1, np.nan)
plt.imshow(mask_overlay, cmap="autumn", alpha=0.7, origin="lower")

plt.title(f"{subject_name} — slice {slice_with_lesion} (microbleed in red)")
plt.axis("off")
plt.savefig("../../reports/sub-101_slice_preview.png", dpi=150, bbox_inches="tight")
print("Saved image to reports/sub-101_slice_preview.png — open it in VS Code's file explorer!")
plt.show()