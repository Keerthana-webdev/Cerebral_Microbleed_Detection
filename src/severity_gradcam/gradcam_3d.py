"""
Lesson 12b: A Grad-CAM implementation for our 3D CNN. This is written
manually (not using an off-the-shelf library) because most Grad-CAM
libraries assume 2D images — ours works on 3D brain patches.
"""

import torch
import torch.nn.functional as F


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        # Hook: capture the layer's output every time we run a forward pass
        target_layer.register_forward_hook(self._save_activations)
        # Hook: capture the gradients flowing back through that layer
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, patch_tensor):
        """
        patch_tensor: shape (1, 1, D, H, W) — a single patch, batch size 1
        Returns: a heatmap the same size as the input patch, values 0-1
        """
        self.model.eval()
        output = self.model(patch_tensor)  # forward pass, also triggers the hook

        self.model.zero_grad()
        output.backward()  # backward pass, triggers the gradient hook

        # Average the gradients across spatial dimensions -> one importance
        # weight per feature-map channel (this is the core Grad-CAM idea)
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)

        # Weighted sum of activation maps, then ReLU (we only care about
        # features that POSITIVELY influenced the "microbleed" decision)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize the heatmap up to match the original patch size
        cam = F.interpolate(cam, size=patch_tensor.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1 for easy visualization
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam