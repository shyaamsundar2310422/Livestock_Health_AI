"""
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability module.
"""
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CLASS_KEYS, CLASS_DISPLAY_NAMES
from src.model import load_model, preprocess_image, get_device


class GradCAM:
    """
    Grad-CAM implementation using forward and full backward hooks.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self.activations = out

    def _save_gradient(self, module: nn.Module, grad_inp: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_out[0]

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Generate 2D activation map (normalized to [0, 1]) for the given input tensor.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        score = output[:, target_class]
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("Failed to capture activations or gradients during Grad-CAM.")

        # Pool gradients across spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class

    def remove_hooks(self) -> None:
        """Safely remove forward and backward hooks."""
        try:
            self._fwd_hook.remove()
        except Exception:
            pass
        try:
            self._bwd_hook.remove()
        except Exception:
            pass


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the appropriate convolutional feature layer for Grad-CAM.
    - DenseNet-121: final dense block layer (features.denseblock4.denselayer16.conv2)
    - EfficientNet-B0: final convolutional stage (features[-1])
    """
    if model_name == "densenet121":
        return model.features.denseblock4.denselayer16.conv2
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Grad-CAM target layer not configured for model: {model_name}")


def generate_gradcam_visualizations(
    image: Image.Image,
    model_name: str = "densenet121",
    model: Optional[nn.Module] = None,
    target_class: Optional[int] = None,
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Generate Grad-CAM heatmap and overlay images on top of the original PIL Image.

    Returns:
        dict: {
            "heatmap": PIL.Image,
            "overlay": PIL.Image,
            "cam": np.ndarray (float in [0, 1]),
            "target_class_idx": int,
            "target_class_name": str
        }
    """
    if device is None:
        device = get_device()
    if model is None:
        model = load_model(model_name=model_name, device=device)

    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    try:
        input_tensor = preprocess_image(image).to(device)
        cam, pred_class_idx = grad_cam.generate(input_tensor, target_class=target_class)

        orig_w, orig_h = image.size
        # Resize CAM to match original image size
        cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Apply JET colormap
        heatmap_u8 = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend with original image
        img_np = np.array(image.convert("RGB"))
        alpha_clamped = max(0.0, min(1.0, float(alpha)))
        overlay_np = cv2.addWeighted(img_np, 1.0 - alpha_clamped, heatmap_rgb, alpha_clamped, 0)

        target_key = CLASS_KEYS[pred_class_idx]
        target_display = CLASS_DISPLAY_NAMES[target_key]

        return {
            "heatmap": Image.fromarray(heatmap_rgb),
            "overlay": Image.fromarray(overlay_np),
            "cam": cam_resized,
            "target_class_idx": pred_class_idx,
            "target_class_name": target_display,
        }
    finally:
        grad_cam.remove_hooks()
