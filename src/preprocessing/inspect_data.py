"""
Lesson 2: Load ONE real brain scan and just look at it — no AI yet.
This just proves we can read the files correctly.
"""

import nibabel as nib      # the library that reads .nii.gz files
from config import RAW_DATA_DIR, SWI_PATTERN, MASK_PATTERN

# Pick the first subject to inspect
subject_name = "sub-101"
subject_folder = RAW_DATA_DIR / subject_name

swi_file = subject_folder / SWI_PATTERN.format(subj=subject_name)
mask_file = subject_folder / MASK_PATTERN.format(subj=subject_name)

print(f"Looking for: {swi_file}")

# Load the scan
swi_image = nib.load(str(swi_file))
mask_image = nib.load(str(mask_file))

# .shape tells us the size of the 3D cube: (width, height, num_slices)
print("Scan shape (voxels):", swi_image.shape)

# .header.get_zooms() tells us real-world size of each voxel, in millimeters
print("Voxel size (mm):", swi_image.header.get_zooms())

# Turn it into a plain numpy array of numbers so we can do math on it
swi_data = swi_image.get_fdata()
print("Brightness range: min =", swi_data.min(), " max =", swi_data.max())

# Count how many voxels are marked as "microbleed" in the answer key
mask_data = mask_image.get_fdata()
num_lesion_voxels = (mask_data > 0).sum()
print("Number of microbleed voxels marked by radiologist:", int(num_lesion_voxels))