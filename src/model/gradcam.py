"""Grad-CAM attention visualization for ResNet-18 encoder."""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Grad-CAM: backprop from encoder embedding to last conv (ResNet-18)."""

    def __init__(self, encoder: nn.Module, target_layer: Optional[nn.Module] = None):
        self.encoder = encoder
        self.encoder.eval()
        if target_layer is None:
            # ResNet-18: last conv is in layer4
            if hasattr(encoder, "layer4"):
                for m in reversed(list(encoder.layer4.modules())):
                    if isinstance(m, nn.Conv2d):
                        target_layer = m
                        break
            # DenseNet-121: last conv in features
            if target_layer is None and hasattr(encoder, "features"):
                for m in reversed(list(encoder.features.modules())):
                    if isinstance(m, nn.Conv2d):
                        target_layer = m
                        break
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach() if grad_output[0] is not None else None

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        image: (1, C, H, W). Backprop from encoder output (embedding sum) or, if target is int, from logits[0, target] for classifier.
        Returns: (cam heatmap [H,W], raw activations).
        """
        if image.grad is not None:
            image.grad.zero_()
        self.encoder.zero_grad()
        out = self.encoder(image)
        # For classifier (logits [1, C]): backprop from predicted class. Else from sum (encoder embedding).
        if isinstance(target, int) and out.dim() == 2 and out.size(1) > 1:
            scalar = out[0, target]
        else:
            scalar = out.sum()
        scalar.backward()
        act = self.activations[0].cpu().numpy()
        grad = self.gradients
        if grad is None:
            # Fallback: uniform weights
            cam = np.mean(act, axis=0).astype(np.float32)
        else:
            grad = grad[0].cpu().numpy()
            weights = grad.mean(axis=(1, 2))
            cam = np.zeros(act.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * act[i]
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, act

    def generate_heatmap(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Return heatmap resized to size (H, W), 0-255 uint8."""
        cam, _ = self.__call__(image, target)
        cam_uint8 = (cam * 255).astype(np.uint8)
        return cv2.resize(cam_uint8, (size[1], size[0]))
