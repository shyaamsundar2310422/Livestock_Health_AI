"""
DMI (Dry Matter Intake) prediction module and inference scaffolding for
INRAE Dataset 2 (Pooled Expanded Features Extra Trees Regressor).
"""
import os
import pickle
from datetime import datetime
import math
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import pandas as pd

from src.config import (
    DMI_MODEL_PATH,
    DMI_MODEL_NAME,
    DMI_FEATURE_NAMES,
    DMI_FEATURE_METADATA,
    DMI_DISCLAIMER,
)

# In-memory cache for loaded model artifact
_DMI_MODEL_CACHE: Optional[Any] = None


def check_dmi_model_availability(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Check if the trained DMI model artifact exists on disk.
    Supports .joblib and .pkl formats.
    """
    path = model_path or DMI_MODEL_PATH
    alt_pkl = path.with_suffix(".pkl")
    
    found_path = None
    if path.exists():
        found_path = path
    elif alt_pkl.exists():
        found_path = alt_pkl

    is_available = found_path is not None
    return {
        "is_available": is_available,
        "model_path": str(found_path) if is_available else str(path),
        "model_name": DMI_MODEL_NAME,
        "expected_features": DMI_FEATURE_NAMES,
        "feature_count": len(DMI_FEATURE_NAMES),
        "message": (
            f"Trained model artifact found at {found_path}."
            if is_available
            else f"Model artifact required: '{path.name}' (or .pkl) is missing from models/. "
                 f"Trained model package must be provided to activate predictions."
        ),
    }


def validate_dmi_features(features: Dict[str, Any]) -> Tuple[Optional[Dict[str, float]], List[str]]:
    """
    Validate and cast the 9 input features required by the Pooled Expanded model:
    age, parity, DIM, BW, BCS, Milk, PC_Milk, Fat_Milk, Lac_Milk.

    Returns:
        (validated_dict, error_messages)
    """
    errors: List[str] = []
    validated: Dict[str, float] = {}

    for feature_name in DMI_FEATURE_NAMES:
        meta = DMI_FEATURE_METADATA.get(feature_name, {})
        label = meta.get("label", feature_name)
        val = features.get(feature_name)

        if val is None or (isinstance(val, str) and val.strip() == ""):
            errors.append(f"Missing required feature: '{label}' ({feature_name}).")
            continue

        try:
            float_val = float(val)
        except (ValueError, TypeError):
            errors.append(f"Invalid non-numeric value '{val}' for '{label}' ({feature_name}).")
            continue

        if not math.isfinite(float_val):
            errors.append(f"Value for '{label}' ({feature_name}) must be a finite number.")
            continue

        # Biological range validation
        min_val = meta.get("min", 0.0)
        max_val = meta.get("max", 1e6)

        if float_val < min_val:
            errors.append(f"Value {float_val} for '{label}' is below minimum reasonable threshold ({min_val} {meta.get('unit', '')}).")
        elif float_val > max_val:
            errors.append(f"Value {float_val} for '{label}' exceeds maximum reasonable threshold ({max_val} {meta.get('unit', '')}).")
        else:
            validated[feature_name] = float_val

    if errors:
        return None, errors
    return validated, []


def load_dmi_model(model_path: Optional[Path] = None) -> Any:
    """
    Load the trained DMI regressor from disk using joblib or pickle.
    Raises FileNotFoundError if artifact is missing.
    """
    global _DMI_MODEL_CACHE
    if _DMI_MODEL_CACHE is not None:
        return _DMI_MODEL_CACHE

    status = check_dmi_model_availability(model_path)
    if not status["is_available"]:
        raise FileNotFoundError(status["message"])

    resolved_path = Path(status["model_path"])
    
    # Try joblib first, fall back to pickle
    try:
        import joblib
        model = joblib.load(str(resolved_path))
    except Exception:
        with open(resolved_path, "rb") as f:
            model = pickle.load(f)

    _DMI_MODEL_CACHE = model
    return model


def predict_dmi(
    features: Dict[str, Any],
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Predict daily Dry Matter Intake (DMI in kg/day) using the trained Pooled Expanded model.

    Returns structured result dictionary. If model file is missing or inputs are invalid,
    handles gracefully without crashing.
    """
    # 1. Validate features
    validated_dict, errors = validate_dmi_features(features)
    if errors:
        return {
            "status": "error",
            "error_type": "VALIDATION_ERROR",
            "errors": errors,
            "message": "Input validation failed. Please check feature values.",
        }

    # 2. Check model availability
    availability = check_dmi_model_availability(model_path)
    if not availability["is_available"]:
        return {
            "status": "model_required",
            "error_type": "MODEL_ARTIFACT_REQUIRED",
            "message": availability["message"],
            "model_path": availability["model_path"],
            "required_features": DMI_FEATURE_NAMES,
            "validated_inputs": validated_dict,
        }

    # 3. Load model and predict
    try:
        model = load_dmi_model(model_path)
        # Construct DataFrame in exact feature column order
        X = pd.DataFrame([validated_dict], columns=DMI_FEATURE_NAMES)
        pred_value = float(model.predict(X)[0])

        return {
            "status": "success",
            "predicted_dmi_kg_day": round(pred_value, 2),
            "input_features": validated_dict,
            "model_name": DMI_MODEL_NAME,
            "unit": "kg/day",
            "prediction_timestamp": datetime.now().isoformat(timespec="seconds"),
            "disclaimer": DMI_DISCLAIMER,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": "INFERENCE_ERROR",
            "message": f"DMI Inference execution failed: {str(e)}",
        }
