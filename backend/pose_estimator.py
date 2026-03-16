"""
Pose-Based Height/Weight Predictor
===================================
Uses TorchVision's Keypoint R-CNN for pose detection and trained U-Net models
for height and weight prediction.
"""

import numpy as np
import torch
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional

# Import trained model loader
from model_loader import ModelManager

# COCO keypoint indices (from Keypoint R-CNN)
KP = {
    'nose':           0,
    'left_eye':       1, 'right_eye':       2,
    'left_ear':       3, 'right_ear':       4,
    'left_shoulder':  5, 'right_shoulder':  6,
    'left_elbow':     7, 'right_elbow':     8,
    'left_wrist':     9, 'right_wrist':    10,
    'left_hip':      11, 'right_hip':      12,
    'left_knee':     13, 'right_knee':     14,
    'left_ankle':    15, 'right_ankle':    16,
}

# Module-level model cache
_model = None


def load_pose_model():
    """Download (once) and return the pretrained Keypoint R-CNN model."""
    global _model
    if _model is not None:
        return _model
    from torchvision.models.detection import (
        keypointrcnn_resnet50_fpn,
        KeypointRCNN_ResNet50_FPN_Weights,
    )
    print("  Loading Keypoint R-CNN (downloads ~170 MB on first run)…")
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    _model = keypointrcnn_resnet50_fpn(weights=weights, progress=True)
    _model.eval()
    print("  Keypoint R-CNN ready.")
    return _model


def _vis(kps: np.ndarray, name: str) -> bool:
    """Check if keypoint is visible (confidence > 0.5)."""
    return kps[KP[name], 2] > 0.5


def estimate_height_weight(image_bytes: bytes) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    Detect body pose and estimate height/weight using trained models.
    
    Pipeline:
    1. Load and validate image
    2. Run Keypoint R-CNN pose detection for validation
    3. Run trained U-Net models for height/weight prediction
    4. Calculate BMI and return results

    Returns
    -------
    height_cm : float | None
    weight_kg : float | None
    prior_bmi : float | None  — calculated BMI from predictions
    error     : str | None  — human-readable error message
    """
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Load and validate image
    # ──────────────────────────────────────────────────────────────────────────
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = img.size
    except Exception as exc:
        return None, None, None, f"Cannot open image: {exc}"

    # Convert image to tensor - properly normalized
    # PyTorch model expects normalized values in range [0, 1]
    img_array = np.array(img, dtype=np.float32)  # uint8 [0, 255] -> float32 [0, 255]
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Run pose detection
    # ──────────────────────────────────────────────────────────────────────────
    model = load_pose_model()
    with torch.no_grad():
        outputs = model([img_tensor])
    result = outputs[0]

    if len(result["scores"]) == 0 or result["scores"][0].item() < 0.4:
        return (
            None, None, None,
            "No person detected. Please take a clear, well-lit, full-body photo from head to feet."
        )

    kps = result["keypoints"][0].numpy()  # (17, 3): [x, y, visibility]

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Validate critical keypoints are visible
    # ──────────────────────────────────────────────────────────────────────────
    critical = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose"]
    if not all(_vis(kps, lm) for lm in critical):
        return (
            None, None, None,
            "Body parts not fully visible. Please ensure head, shoulders, and hips are visible."
        )

    # Feet required for height
    has_both_feet = _vis(kps, "left_ankle") and _vis(kps, "right_ankle")
    if not has_both_feet:
        return (
            None, None, None,
            "Feet not visible. Please take a full-body photo showing your complete body down to the feet."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Use trained models for height/weight prediction
    # ──────────────────────────────────────────────────────────────────────────
    try:
        # Load trained models if not already loaded
        if not ModelManager.models_loaded():
            ModelManager.load_models()
        
        # Prepare image for model input
        # Models expect 256x256 images in [0, 255] range (uint8 or float32)
        img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)  # [0, 255]
        # DO NOT normalize - models may expect raw pixel values
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
        
        # Move to same device as models
        device = next(ModelManager.get_height_model().parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Run inference with trained models
        with torch.no_grad():
            height_output = ModelManager.get_height_model()(img_tensor)
            weight_output = ModelManager.get_weight_model()(img_tensor)
        
        # Extract scalar values
        # Models output raw values that need transformation:
        # height_raw / -40 = height_cm
        # weight_raw / -275 = weight_kg
        height_raw = float(height_output.squeeze().item())
        weight_raw = float(weight_output.squeeze().item())
        
        # Apply transformation to get actual height and weight
        height_cm = height_raw / -40.0
        weight_kg = weight_raw / -275.0
        
        # Validate predictions are in reasonable ranges
        if height_cm < 0 or weight_kg < 0:
            return (
                None, None, None,
                f"Model prediction validation failed. Please retake the photo in better lighting."
            )
        
        if height_cm < 130 or height_cm > 230:
            return (
                None, None, None,
                f"Height prediction out of typical range ({height_cm:.0f} cm). Please retake the photo in better lighting."
            )
        
        if weight_kg < 40 or weight_kg > 150:
            return (
                None, None, None,
                f"Weight prediction out of typical range ({weight_kg:.0f} kg). Please retake the photo in better lighting."
            )
        
        # Calculate BMI
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2)
        prior_bmi = float(bmi)
        
        return float(height_cm), float(weight_kg), prior_bmi, None
    
    except Exception as exc:
        return (
            None, None, None,
            f"Model prediction error: {exc}"
        )
