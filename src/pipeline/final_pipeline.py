"""
The complete, callable end-to-end detection pipeline:
Raw scan -> preprocessing -> sliding window -> Stage 1 CNN -> NMS ->
Stage 2 CNN -> confidence scoring -> severity grading -> results.

This is the single module the dashboard (Phase 6) will import and call.
"""

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from model import SimpleCNN3D

DEVICE = torch.device("cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)
STAGE1_THRESHOLD = 0.7   # chosen from the FROC sweep (Lesson 14)
STAGE2_THRESHOLD = 0.7
NMS_DISTANCE = 15
TARGET_SPACING = (1.0, 1.0, 1.0)


class CMBDetectionPipeline:
    def __init__(self):
        self.stage1_model = SimpleCNN3D().to(DEVICE)
        self.stage1_model.load_state_dict(
            torch.load(MODELS_DIR / "candidate_detector_best.pt", map_location=DEVICE))
        self.stage1_model.eval()

        self.stage2_model = SimpleCNN3D().to(DEVICE)
        self.stage2_model.load_state_dict(
            torch.load(MODELS_DIR / "mimic_classifier_best.pt", map_location=DEVICE))
        self.stage2_model.eval()

    # ---- Preprocessing (same steps as Lesson 4) ----
    def _fix_uneven_brightness(self, image):
        image = sitk.Cast(image, sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        shrink = 4
        small_image = sitk.Shrink(image, [shrink] * image.GetDimension())
        small_mask = sitk.Shrink(mask, [shrink] * image.GetDimension())
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([20] * 3)
        corrector.Execute(small_image, small_mask)
        log_bias = corrector.GetLogBiasFieldAsImage(image)
        return sitk.Cast(image / sitk.Exp(log_bias), sitk.sitkFloat32)

    def _resample(self, image):
        old_spacing = image.GetSpacing()
        old_size = image.GetSize()
        new_size = [int(round(sz * osp / tsp)) for sz, osp, tsp in zip(old_size, old_spacing, TARGET_SPACING)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(TARGET_SPACING)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        return resampler.Execute(image)

    def preprocess(self, nifti_path):
        image = sitk.ReadImage(str(nifti_path))
        image = self._fix_uneven_brightness(image)
        image = self._resample(image)
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        low, high = np.percentile(array, (0.5, 99.5))
        array = np.clip(array, low, high)
        mean, std = array.mean(), array.std()
        return (array - mean) / (std + 1e-8)

    # ---- Sliding window + two-stage detection + NMS ----
    def _cut_patch(self, volume, center, size):
        half = [s // 2 for s in size]
        slices = tuple(slice(c - h, c + h) for c, h in zip(center, half))
        return volume[slices]

    def _generate_grid(self, shape):
        half = [s // 2 for s in PATCH_SIZE]
        centers = []
        for x in range(half[0], shape[0] - half[0], STRIDE[0]):
            for y in range(half[1], shape[1] - half[1], STRIDE[1]):
                for z in range(half[2], shape[2] - half[2], STRIDE[2]):
                    centers.append((x, y, z))
        return centers

    def _apply_nms(self, centers, probs, min_distance):
        order = np.argsort(-np.array(probs))
        kept_centers, kept_probs = [], []
        for i in order:
            c = centers[i]
            too_close = any(np.linalg.norm(np.array(c) - np.array(kc)) < min_distance for kc in kept_centers)
            if not too_close:
                kept_centers.append(c)
                kept_probs.append(probs[i])
        return kept_centers, kept_probs

    def detect(self, volume, batch_size=64, progress_callback=None):
        """Runs the full pipeline on a preprocessed 3D volume. Returns a list
        of {center, confidence, severity_contribution} detections."""
        all_centers = self._generate_grid(volume.shape)
        candidate_centers = [c for c in all_centers if self._cut_patch(volume, c, PATCH_SIZE).std() > 0.05]

        # Stage 1
        stage1_hits = []
        for i in range(0, len(candidate_centers), batch_size):
            batch = candidate_centers[i:i + batch_size]
            patches = np.stack([self._cut_patch(volume, c, PATCH_SIZE) for c in batch])
            tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                probs = torch.sigmoid(self.stage1_model(tensor)).squeeze(1).numpy()
            stage1_hits.extend([(c, p) for c, p in zip(batch, probs) if p > STAGE1_THRESHOLD])
            if progress_callback:
                progress_callback(min(1.0, i / max(len(candidate_centers), 1)) * 0.5)

        # Stage 2
        final = []
        for i in range(0, len(stage1_hits), batch_size):
            batch = stage1_hits[i:i + batch_size]
            batch_centers = [c for c, p in batch]
            patches = np.stack([self._cut_patch(volume, c, PATCH_SIZE) for c in batch_centers])
            tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                probs = torch.sigmoid(self.stage2_model(tensor)).squeeze(1).numpy()
            for c, p in zip(batch_centers, probs):
                if p > STAGE2_THRESHOLD:
                    final.append((c, float(p)))
            if progress_callback:
                progress_callback(0.5 + min(1.0, i / max(len(stage1_hits), 1)) * 0.5)

        # NMS
        if final:
            centers, probs = zip(*final)
            kept_centers, kept_probs = self._apply_nms(list(centers), list(probs), NMS_DISTANCE)
        else:
            kept_centers, kept_probs = [], []

        detections = []
        for center, prob in zip(kept_centers, kept_probs):
            confidence_label = (
                "REVIEW RECOMMENDED" if 0.35 <= prob <= 0.65
                else ("High confidence" if prob > 0.65 else "Low confidence")
            )
            detections.append({
                "center": center, "confidence": round(prob, 3), "confidence_label": confidence_label,
            })
        return detections

    def grade_severity(self, num_detections):
        if num_detections == 0:
            return "None"
        elif num_detections <= 2:
            return "Mild"
        elif num_detections <= 5:
            return "Moderate"
        else:
            return "Severe"