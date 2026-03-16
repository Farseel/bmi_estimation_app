"""
Model Loader Module

Loads custom U-Net models for height and weight prediction.
Architecture reconstructed from saved model checkpoint key/shape inspection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional

HEIGHT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_ep_48.pth")
WEIGHT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_ep_37.pth")

# Lite (float16, state-dict-only) versions produced by compress_models.py
# DISABLED: Using full models for better accuracy
HEIGHT_LITE_PATH = None  # os.path.join(os.path.dirname(__file__), "..", "models", "height_model_lite.pth")
WEIGHT_LITE_PATH = None  # os.path.join(os.path.dirname(__file__), "..", "models", "weight_model_lite.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Encoder block: double conv with ReLU (no pooling inside)
# ─────────────────────────────────────────────────────────────
class ConvDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),   # index 0
            nn.ReLU(inplace=True),                     # index 1
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # index 2
            nn.ReLU(inplace=True),                     # index 3
        )

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────────────────────
# Decoder block: upsample → conv → concat skip → double conv
# ─────────────────────────────────────────────────────────────
class ConvUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # conv1: bilinear upsample (no params) then conv to halve channels
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # index 0
            nn.Conv2d(in_ch, out_ch, 3, padding=1),   # index 1
        )
        # conv2: double conv after concat with skip connection (out_ch*2 input)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),  # index 0
            nn.ReLU(inplace=True),                          # index 1
            nn.Conv2d(out_ch, out_ch, 3, padding=1),       # index 2
            nn.ReLU(inplace=True),                          # index 3
        )

    def forward(self, x, skip):
        x = self.conv1(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(x)


# ─────────────────────────────────────────────────────────────
# Output heads (height model)
# ─────────────────────────────────────────────────────────────
class ConvOutHeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_out  = nn.Conv2d(128, 2,  1)
        self.joint_out = nn.Conv2d(128, 19, 1)
        self.height_1  = nn.Sequential(nn.Conv2d(128, 32, 1))          # index 0
        self.height_2  = nn.Sequential(
            nn.Linear(32768, 1024),   # index 0
            nn.ReLU(inplace=True),    # index 1
            nn.Dropout(0.5),          # index 2
            nn.Linear(1024, 1),       # index 3
        )

    def forward(self, x):
        h = self.height_1(x)
        # Ensure spatial size is 32×32 regardless of input image size
        if h.shape[-1] != 32 or h.shape[-2] != 32:
            h = F.adaptive_avg_pool2d(h, (32, 32))
        h = h.flatten(1)        # (batch, 32768)
        return self.height_2(h)  # (batch, 1)


# ─────────────────────────────────────────────────────────────
# Output heads (weight model – shares height heads, adds weight heads)
# ─────────────────────────────────────────────────────────────
class ConvOutWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_out  = nn.Conv2d(128, 2,  1)
        self.joint_out = nn.Conv2d(128, 19, 1)
        self.height_1  = nn.Sequential(nn.Conv2d(128, 32, 1))
        self.height_2  = nn.Sequential(
            nn.Linear(32768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )
        self.weight_1  = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1))  # 3×3 kernel
        self.weight_2  = nn.Sequential(
            nn.Linear(32768, 1024),   # index 0
            nn.ReLU(inplace=True),    # index 1
            nn.Dropout(0.5),          # index 2
            nn.Linear(1024, 1),       # index 3
        )

    def forward(self, x):
        w = self.weight_1(x)
        if w.shape[-1] != 32 or w.shape[-2] != 32:
            w = F.adaptive_avg_pool2d(w, (32, 32))
        w = w.flatten(1)
        return self.weight_2(w)


# ─────────────────────────────────────────────────────────────
# Full U-Net models
# Conv_down blocks do NOT include pooling; the main forward does
# MaxPool2d(2) between levels and stores skip connections
# ─────────────────────────────────────────────────────────────
class HeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_down1    = ConvDown(3,    128)
        self.conv_down2    = ConvDown(128,  256)
        self.conv_down3    = ConvDown(256,  512)
        self.conv_down4    = ConvDown(512,  1024)
        self.conv_down5    = ConvDown(1024, 2048)
        self.conv_upsample1 = ConvUpsample(2048, 1024)
        self.conv_upsample2 = ConvUpsample(1024, 512)
        self.conv_upsample3 = ConvUpsample(512,  256)
        self.conv_upsample4 = ConvUpsample(256,  128)
        self.conv_out       = ConvOutHeight()

    def forward(self, x):
        e1 = self.conv_down1(x)
        e2 = self.conv_down2(F.max_pool2d(e1, 2))
        e3 = self.conv_down3(F.max_pool2d(e2, 2))
        e4 = self.conv_down4(F.max_pool2d(e3, 2))
        e5 = self.conv_down5(F.max_pool2d(e4, 2))
        d1 = self.conv_upsample1(e5, e4)
        d2 = self.conv_upsample2(d1, e3)
        d3 = self.conv_upsample3(d2, e2)
        d4 = self.conv_upsample4(d3, e1)
        return self.conv_out(d4)


class WeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_down1    = ConvDown(3,    128)
        self.conv_down2    = ConvDown(128,  256)
        self.conv_down3    = ConvDown(256,  512)
        self.conv_down4    = ConvDown(512,  1024)
        self.conv_down5    = ConvDown(1024, 2048)
        self.conv_upsample1 = ConvUpsample(2048, 1024)
        self.conv_upsample2 = ConvUpsample(1024, 512)
        self.conv_upsample3 = ConvUpsample(512,  256)
        self.conv_upsample4 = ConvUpsample(256,  128)
        self.conv_out       = ConvOutWeight()

    def forward(self, x):
        e1 = self.conv_down1(x)
        e2 = self.conv_down2(F.max_pool2d(e1, 2))
        e3 = self.conv_down3(F.max_pool2d(e2, 2))
        e4 = self.conv_down4(F.max_pool2d(e3, 2))
        e5 = self.conv_down5(F.max_pool2d(e4, 2))
        d1 = self.conv_upsample1(e5, e4)
        d2 = self.conv_upsample2(d1, e3)
        d3 = self.conv_upsample3(d2, e2)
        d4 = self.conv_upsample4(d3, e1)
        return self.conv_out(d4)


# ─────────────────────────────────────────────────────────────
# Model manager
# ─────────────────────────────────────────────────────────────
def _load_state_dict(full_path: str, lite_path: Optional[str], label: str):
    """Load state_dict from lite file if available, else from full checkpoint."""
    if lite_path is not None and os.path.exists(lite_path):
        print(f"  Using lite model: {os.path.basename(lite_path)}")
        state_dict = torch.load(lite_path, map_location=DEVICE)
        # Convert float16 weights back to float32 for stable CPU inference
        return {k: v.float() for k, v in state_dict.items()}

    print(f"  Loading full checkpoint...")  
    checkpoint = torch.load(full_path, map_location=DEVICE)
    
    # Determine what format the checkpoint is in
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            print(f"    Found state_dict in checkpoint")
            return checkpoint['state_dict']
        else:
            # Check if it looks like state_dict (has module names with dots)
            keys = list(checkpoint.keys())
            if keys and isinstance(keys[0], str) and ('conv' in keys[0] or 'weight' in keys[0]):
                print(f"    Treating checkpoint as direct state_dict")
                return checkpoint
    
    # Otherwise return as-is
    print(f"    Using checkpoint directly as state_dict")
    return checkpoint


class ModelManager:
    _height_model = None
    _weight_model = None

    @classmethod
    def load_models(cls):
        print(f"Loading height model from: {HEIGHT_MODEL_PATH}")
        state_dict = _load_state_dict(HEIGHT_MODEL_PATH, HEIGHT_LITE_PATH, 'height')
        cls._height_model = HeightModel()
        missing, unexpected = cls._height_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  ⚠ Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        cls._height_model.to(DEVICE)
        cls._height_model.eval()
        print("✓ Height model loaded successfully")

        print(f"Loading weight model from: {WEIGHT_MODEL_PATH}")
        state_dict2 = _load_state_dict(WEIGHT_MODEL_PATH, WEIGHT_LITE_PATH, 'weight')
        cls._weight_model = WeightModel()
        missing2, unexpected2 = cls._weight_model.load_state_dict(state_dict2, strict=False)
        if missing2:
            print(f"  ⚠ Missing keys ({len(missing2)}): {missing2[:3]}...")
        if unexpected2:
            print(f"  ⚠ Unexpected keys ({len(unexpected2)}): {unexpected2[:3]}...")
        cls._weight_model.to(DEVICE)
        cls._weight_model.eval()
        print("✓ Weight model loaded successfully")

    @classmethod
    def get_height_model(cls):
        if cls._height_model is None:
            raise RuntimeError("Height model not loaded. Call load_models() first.")
        return cls._height_model

    @classmethod
    def get_weight_model(cls):
        if cls._weight_model is None:
            raise RuntimeError("Weight model not loaded. Call load_models() first.")
        return cls._weight_model

    @classmethod
    def models_loaded(cls):
        return cls._height_model is not None and cls._weight_model is not None

