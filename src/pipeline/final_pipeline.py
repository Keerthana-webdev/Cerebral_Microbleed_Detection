import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
STAGE1_MODEL_PATH = MODEL_DIR / "candidate_detector_v2_best.pt"
STAGE2_MODEL_PATH = MODEL_DIR / "mimic_classifier_best.pt"

sys.path.append(str(PROJECT_ROOT / "src" / "candidate_detector"))

from model import SimpleCNN3D
DEVICE = torch.device("cpu")
PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)
STAGE1_THRESHOLD = 0.60
STAGE2_THRESHOLD = 0.70
NMS_DISTANCE = 20
TARGET_SPACING = (1.0, 1.0, 1.0)

class CMBDetectionPipeline:

    def __init__(self):
        self.stage1_model = SimpleCNN3D().to(DEVICE)
        self.stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE))
        self.stage1_model.eval()
        self.stage2_model = SimpleCNN3D().to(DEVICE)
        self.stage2_model.load_state_dict( torch.load(STAGE2_MODEL_PATH, map_location=DEVICE))
        self.stage2_model.eval()

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

        corrected = image / sitk.Exp(log_bias)

        return sitk.Cast(corrected, sitk.sitkFloat32)

    def _resample(self, image):

        old_spacing = image.GetSpacing()
        old_size = image.GetSize()

        new_size = [
            int(round(size * old_sp / target_sp))
            for size, old_sp, target_sp in zip(old_size, old_spacing, TARGET_SPACING)
        ]

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

        mean = array.mean()
        std = array.std()

        array = (array - mean) / (std + 1e-8)

        return array

    def _cut_patch(self, volume, center, size):

        half = [s // 2 for s in size]

        slices = tuple(
            slice(c - h, c + h)
            for c, h in zip(center, half)
        )

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

        order = np.argsort(-np.asarray(probs))

        kept_centers = []
        kept_probs = []

        for index in order:
            center = centers[index]

            too_close = any(
                np.linalg.norm(np.asarray(center) - np.asarray(kept)) < min_distance
                for kept in kept_centers
            )

            if not too_close:
                kept_centers.append(center)
                kept_probs.append(probs[index])

        return kept_centers, kept_probs


    def detect(self, volume, batch_size=64, progress_callback=None):

        all_centers = self._generate_grid(volume.shape)

        candidate_centers = []
        for center in all_centers:
            patch = self._cut_patch(volume, center, PATCH_SIZE)
            if patch.shape != PATCH_SIZE:
                continue
            if patch.std() > 0.05:
                candidate_centers.append(center)

        stage1_hits = []
        total_candidates = len(candidate_centers)

        for start in range(0, total_candidates, batch_size):
            batch_centers = candidate_centers[start:start + batch_size]

            patches = np.stack([
                self._cut_patch(volume, center, PATCH_SIZE)
                for center in batch_centers
            ])

            tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                logits = self.stage1_model(tensor)
                probs = torch.sigmoid(logits).reshape(-1).cpu().numpy()

            for center, prob in zip(batch_centers, probs):
                if prob > STAGE1_THRESHOLD:
                    stage1_hits.append((center, float(prob)))

            if progress_callback:
                fraction = start / max(total_candidates, 1)
                progress_callback(fraction * 0.50)

        final_candidates = []
        total_stage1 = len(stage1_hits)

        for start in range(0, total_stage1, batch_size):
            batch = stage1_hits[start:start + batch_size]
            batch_centers = [center for center, _ in batch]

            patches = np.stack([
                self._cut_patch(volume, center, PATCH_SIZE)
                for center in batch_centers
            ])

            tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                logits = self.stage2_model(tensor)
                probs = torch.sigmoid(logits).reshape(-1).cpu().numpy()

            for center, prob in zip(batch_centers, probs):
                if prob > STAGE2_THRESHOLD:
                    final_candidates.append((center, float(prob)))

            if progress_callback:
                fraction = start / max(total_stage1, 1)
                progress_callback(0.50 + fraction * 0.50)

        if final_candidates:
            centers, probs = zip(*final_candidates)
            kept_centers, kept_probs = self._apply_nms(list(centers), list(probs), NMS_DISTANCE)
        else:
            kept_centers, kept_probs = [], []

        detections = []

        for index, (center, probability) in enumerate(zip(kept_centers, kept_probs), start=1):

            if 0.35 <= probability <= 0.65:
                confidence_label = "REVIEW RECOMMENDED"
            elif probability > 0.65:
                confidence_label = "High confidence"
            else:
                confidence_label = "Low confidence"

            detections.append({
                "id": f"CMB-{index:02d}",
                "center": tuple(int(v) for v in center),
                "confidence": round(probability, 3),
                "confidence_label": confidence_label,
                "classification": (
                    "True microbleed" if probability >= STAGE2_THRESHOLD else "Mimic"
                ),
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

    def generate_gradcam_for_detection(self, volume, center):
        """
        Returns (patch, heatmap) for one detected candidate location.
        patch: the raw 3D patch (16, 16, 8), for display.
        heatmap: same shape, values 0-1, showing model attention.
        """
        severity_gradcam_path = str(PROJECT_ROOT / "src" / "severity_gradcam")
        if severity_gradcam_path not in sys.path:
            sys.path.append(severity_gradcam_path)

        from gradcam_utils import generate_gradcam

        patch = self._cut_patch(volume, center, PATCH_SIZE)
        heatmap = generate_gradcam(self.stage2_model, patch, positive_class=True)
        return patch, heatmap