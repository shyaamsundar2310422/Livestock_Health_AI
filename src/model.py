"""
Model loading, preprocessing, and inference routines.
"""
from typing import Dict, Any, Tuple, Optional
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from src.config import (
    CLASS_KEYS,
    CLASS_DISPLAY_NAMES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DENSENET121_WEIGHTS,
    EFFICIENTNET_WEIGHTS,
)

# Standard evaluation preprocessing transform matching training notebooks
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_densenet121(weights_path: Optional[Path] = None, device: Optional[torch.device] = None) -> nn.Module:
    """
    Build DenseNet-121 architecture with 3-class linear head
    and load weights from checkpoint.
    """
    if device is None:
        device = get_device()
    weights_path = weights_path or DENSENET121_WEIGHTS
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"DenseNet-121 weights not found at: {weights_path}")

    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(CLASS_KEYS))

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def build_efficientnet_b0(weights_path: Optional[Path] = None, device: Optional[torch.device] = None) -> nn.Module:
    """
    Build EfficientNet-B0 architecture with 3-class linear head
    and load weights from checkpoint.
    """
    if device is None:
        device = get_device()
    weights_path = weights_path or EFFICIENTNET_WEIGHTS
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"EfficientNet-B0 weights not found at: {weights_path}")

    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_KEYS))

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# Global model cache to avoid reloading weights repeatedly
_MODEL_CACHE: Dict[str, nn.Module] = {}


def load_model(model_name: str = "densenet121", device: Optional[torch.device] = None) -> nn.Module:
    """
    Load and cache the requested model ('densenet121' or 'efficientnet_b0').
    """
    if device is None:
        device = get_device()
    cache_key = f"{model_name}_{device.type}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if model_name == "densenet121":
        model = build_densenet121(device=device)
    elif model_name == "efficientnet_b0":
        model = build_efficientnet_b0(device=device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose 'densenet121' or 'efficientnet_b0'.")

    _MODEL_CACHE[cache_key] = model
    return model


def validate_and_load_image(image_file_or_path: Any) -> Image.Image:
    """
    Validate and open an image file or stream, returning an RGB PIL Image.
    Raises ValueError on invalid or unreadable images.
    """
    try:
        if isinstance(image_file_or_path, Image.Image):
            img = image_file_or_path
        else:
            img = Image.open(image_file_or_path)

        # Force load image data to catch corrupt file issues early
        img.load()
        rgb_img = img.convert("RGB")
        w, h = rgb_img.size
        if w < 10 or h < 10:
            raise ValueError(f"Image dimensions too small ({w}x{h}). Please provide a clear image.")
        return rgb_img
    except UnidentifiedImageError:
        raise ValueError("The provided file could not be recognized as a valid image format.")
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image into a normalized tensor with batch dimension: [1, 3, 224, 224].
    """
    tensor = EVAL_TRANSFORM(image)
    return tensor.unsqueeze(0)


def predict(
    image: Image.Image,
    model_name: str = "densenet121",
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run disease prediction on a given PIL Image.

    Returns:
        dict: {
            "predicted_key": str,
            "predicted_class": str,
            "confidence": float (0.0 to 1.0),
            "confidence_percent": str ("XX.XX%"),
            "probabilities": {class_name: float},
            "raw_scores": [float],
            "model_name": str
        }
    """
    if device is None:
        device = get_device()
    if model is None:
        model = load_model(model_name=model_name, device=device)

    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_key = CLASS_KEYS[pred_idx]
    pred_display = CLASS_DISPLAY_NAMES[pred_key]
    confidence = float(probs[pred_idx])

    probabilities = {
        CLASS_DISPLAY_NAMES[k]: float(probs[i])
        for i, k in enumerate(CLASS_KEYS)
    }

    return {
        "predicted_key": pred_key,
        "predicted_class": pred_display,
        "confidence": confidence,
        "confidence_percent": f"{confidence * 100:.2f}%",
        "probabilities": probabilities,
        "raw_probs": probs.tolist(),
        "model_name": model_name,
    }
