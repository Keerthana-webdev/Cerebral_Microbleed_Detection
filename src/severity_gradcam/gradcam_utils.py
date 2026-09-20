"""
CMB Detection Project
Grad-CAM helper for the Stage-2 mimic classifier.

This module creates a model-attention heatmap for a selected candidate patch.
It automatically finds the last Conv3d layer in the existing CNN.
"""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


DEVICE = torch.device("cpu")


def _find_last_conv3d(model):
    """
    Automatically find the last Conv3d layer.
    This avoids hard-coding a layer name from the existing model.
    """
    last_conv = None

    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            last_conv = module

    if last_conv is None:
        raise RuntimeError("No Conv3d layer found in the model.")

    return last_conv


def generate_gradcam(model, patch, positive_class=True):
    """
    Generate a Grad-CAM heatmap for a single 3D patch.

    Parameters
    ----------
    model:
        Existing Stage-2 CNN.

    patch:
        numpy array shaped (X, Y, Z).

    positive_class:
        True means explain the microbleed score.

    Returns
    -------
    heatmap:
        numpy array shaped (X, Y, Z), normalized 0-1.
    """

    model.eval()

    if patch.ndim != 3:
        raise ValueError(
            f"Expected 3D patch, received shape {patch.shape}"
        )

    tensor = torch.tensor(
        patch,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    target_layer = _find_last_conv3d(model)

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        model.zero_grad()

        output = model(tensor)

        # Existing binary CNN normally returns:
        # [batch, 1]
        if output.ndim == 2:
            score = output[:, 0].sum()
        else:
            score = output.sum()

        if not positive_class:
            score = -score

        score.backward()

        activation = activations[0]
        gradient = gradients[0]

        # Global average pooling of gradients
        weights = gradient.mean(
            dim=(2, 3, 4),
            keepdim=True
        )

        cam = (weights * activation).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        # Resize heatmap back to patch dimensions
        cam = F.interpolate(
            cam,
            size=patch.shape,
            mode="trilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize
        cam -= cam.min()

        max_value = cam.max()

        if max_value > 1e-8:
            cam /= max_value

        return cam

    finally:

        forward_handle.remove()
        backward_handle.remove()


def create_attention_overlay(
    image_slice,
    heatmap_slice,
    alpha=0.45
):
    """
    Return an RGB overlay suitable for Streamlit display.
    """

    image = np.asarray(image_slice, dtype=np.float32)

    # Normalize image
    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    image = np.clip(
        (image - low) / (high - low + 1e-8),
        0,
        1
    )

    gray_rgb = np.stack(
        [image, image, image],
        axis=-1
    )

    heat = np.asarray(
        heatmap_slice,
        dtype=np.float32
    )

    heat = np.clip(
        heat,
        0,
        1
    )

    # Simple red heatmap
    heat_rgb = np.zeros(
        (*heat.shape, 3),
        dtype=np.float32
    )

    heat_rgb[..., 0] = heat
    heat_rgb[..., 1] = heat * 0.25

    overlay = (
        (1 - alpha) * gray_rgb
        + alpha * heat_rgb
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return overlay