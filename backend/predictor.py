"""
Prediction Module
=================
Estimates height, weight, and BMI from a raw image using
TorchVision's Keypoint R-CNN for pose detection and
anthropometric geometric analysis.
"""

from typing import Tuple, Optional
from pose_estimator import estimate_height_weight


def predict_from_image(image_bytes: bytes) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    Full prediction pipeline from raw image bytes.

    Returns
    -------
    height_cm : float | None
    weight_kg : float | None
    prior_bmi : float | None  — BMI before clipping
    error_msg : str | None  — set when detection fails
    """
    height_cm, weight_kg, prior_bmi, error = estimate_height_weight(image_bytes)
    return height_cm, weight_kg, prior_bmi, error


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """BMI = weight_kg / height_m^2"""
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 2)


def get_bmi_category(bmi: float) -> str:
    """WHO BMI categories."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"
