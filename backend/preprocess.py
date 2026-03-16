"""
Image Preprocessing Module

Converts uploaded images to properly normalized PyTorch tensors
for the U-Net height/weight prediction model.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

# Standard ImageNet normalization values used during training
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# 256x256 is the model's training input size. The U-Net encoder has 4
# max-pool layers, producing 16x16 feature maps at the bottleneck.
# After 4 upsample blocks the final feature map is 256x256. The
# height/weight output heads apply adaptive_avg_pool to 32x32 to match
# the linear layer input of 32*32*32 = 32768.
IMAGE_SIZE = 256


def preprocess_image(image_bytes) -> torch.Tensor:
    """
    Preprocess an uploaded image for model inference.

    Resizes to 256x256, converts to RGB, applies ImageNet normalization,
    and adds a batch dimension.

    Args:
        image_bytes: Raw image file bytes

    Returns:
        torch.Tensor: shape (1, 3, 256, 256)

    Raises:
        ValueError: If image cannot be processed
    """
    try:
        image = Image.open(BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

        tensor = transform(image)
        return tensor.unsqueeze(0)  # (1, 3, 256, 256)

    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

