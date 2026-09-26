"""
Lesson 26: Cache whole-volume sliding-window predictions from CNN v3,
ResNet, and DenseNet on the VALIDATION set, so we can test whether
ensembling helps whole-MRI false-positive control. Test set untouched.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))

from config import PROCESSED_DATA_DIR, SPLITS_DIR
from model_v3 import DeeperCNN3D
from src.models.resnet3d_model import ResNet3D
from src.models.densenet3d_model import DenseNet3D

DEVICE = torch.device("cpu")
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "data" / "ensemble_cache_val"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)

# ---- Load all three models ----
cnn_model = DeeperCNN3D().to(DEVICE)
cnn_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_v3_best.pt", map_location=DEVICE))
cnn_model.eval()

resnet_model = ResNet3D(num_classes=2).to(DEVICE)
resnet_ckpt = torch.load(MODELS_DIR / "resnet3d" / "resnet3d_best.pt", map_location=DEVICE)
resnet_model.load_state_dict(resnet_ckpt["model_state_dict"])
resnet_model.eval()

densenet_model = DenseNet3D(num_classes=2).to(DEVICE)
densenet_ckpt = torch.load(MODELS_DIR / "densenet3d" / "densenet3d_best.pt", map_location=DEVICE)
if isinstance(densenet_ckpt, dict) and "model_state_dict" in densenet_ckpt:
    densenet_model.load_state_dict(densenet_ckpt["model_state_dict"])
else:
    densenet_model.load_state_dict(densenet_ckpt)
densenet_model.eval()

print("All three models loaded.\n")


def generate_grid_centers(volume_shape, patch_size, stride):
    half = [s // 2 for s in patch_size]
    centers = []
    for x in range(half[0], volume_shape[0] - half[0], stride[0]):
        for y in range(half[1], volume_shape[1] - half[1], stride[1]):
            for z in range(half[2], volume_shape[2] - half[2], stride[2]):
                centers.append((x, y, z))
    return centers


def cut_patch(volume, center, size):
    half = [s // 2 for s in size]
    slices = tuple(slice(c - h, c + h) for c, h in zip(center, half))
    return volume[slices]


def process_subject(subject_name, batch_size=64):
    volume = np.load(PROCESSED_DATA_DIR / subject_name / "swi.npy")
    all_centers = generate_grid_centers(volume.shape, PATCH_SIZE, STRIDE)
    candidate_centers = [c for c in all_centers if cut_patch(volume, c, PATCH_SIZE).std() > 0.05]
    print(f"  {subject_name}: {len(candidate_centers)} candidate positions")

    cnn_probs, resnet_probs, densenet_probs = [], [], []

    for i in range(0, len(candidate_centers), batch_size):
        batch_centers = candidate_centers[i:i + batch_size]
        patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in batch_centers])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            cnn_out = torch.sigmoid(cnn_model(tensor)).squeeze(1).numpy()
            resnet_out = F.softmax(resnet_model(tensor), dim=1)[:, 1].numpy()
            densenet_out = F.softmax(densenet_model(tensor), dim=1)[:, 1].numpy()

        cnn_probs.extend(cnn_out.tolist())
        resnet_probs.extend(resnet_out.tolist())
        densenet_probs.extend(densenet_out.tolist())

    np.savez(
        CACHE_DIR / f"{subject_name}.npz",
        centers=np.array(candidate_centers),
        cnn_probs=np.array(cnn_probs),
        resnet_probs=np.array(resnet_probs),
        densenet_probs=np.array(densenet_probs),
    )


def main():
    with open(SPLITS_DIR / "splits.json") as f:
        splits = json.load(f)
    val_subjects = splits["val"]

    print(f"Caching ensemble predictions for {len(val_subjects)} VALIDATION subjects...\n")
    for subject_name in val_subjects:
        process_subject(subject_name)

    print(f"\nAll ensemble validation predictions cached to {CACHE_DIR}")
    print("Test set untouched.")


if __name__ == "__main__":
    main()