"""Unit tests for utils/projection.py.

Tests:
- project_world_to_image: Projection from (x, y) world coordinates to (u, v) image coordinates.
- normalize_image_coords: Normalize image coordinates to [-1, 1] range for grid_sample.
- bilinear_sample: Bilinear interpolation sampling from feature maps.

Tested with Identity intrinsic and extrinsic matrices for simplicity.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from utils.projection import (
    project_world_to_image,
    normalize_image_coords,
    bilinear_sample,
)

# Add parent directory to path for clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_camera_downward_extrinsic(device="cpu"):
    E = torch.eye(4, device=device)

    # Correct Rotation Matrix
    R = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=torch.float32,
        device=device,
    )

    E[:3, :3] = R
    E[2, 3] = -1.0  # 1m above ground
    return E


def test_project_world_to_image():
    B, Q = 1, 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    world_coords = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32, device=device
    )

    K = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    E = get_camera_downward_extrinsic(device).unsqueeze(0)

    img_coords = project_world_to_image(world_coords, K, E)

    assert torch.allclose(
        img_coords[0, 0], torch.tensor([0.0, 0.0], device=device), atol=1e-4
    )
    assert torch.allclose(
        img_coords[0, 1], torch.tensor([1.0, 0.0], device=device), atol=1e-4
    )

    print("test_project_world_to_image passed!")


def test_normalize_image_coords():
    """Test normalization of image coordinates to [-1, 1].

    Check that:
    - (0,0) pixel maps to (-1,-1)
    - (W-1,H-1) pixel maps to (1,1)
    """
    B, Q = 1, 2
    img_coords = torch.tensor([[[0.0, 0.0], [1279.0, 719.0]]])  # (B, Q, 2)
    img_size = (720, 1280)  # (H, W)

    norm_coords = normalize_image_coords(img_coords, img_size)

    # Check shapes
    assert norm_coords.shape == (B, Q, 2)
    # (0,0) -> (-1, -1)
    assert torch.allclose(norm_coords[0, 0], torch.tensor([-1.0, -1.0]), atol=1e-5)
    # (1279,719) -> (1, 1)
    assert torch.allclose(norm_coords[0, 1], torch.tensor([1.0, 1.0]), atol=1e-5)

    print("test_normalize_image_coords passed!")


def test_bilinear_sample():
    """Test bilinear sampling from feature map.

    We create a dummy 4x4 feature map and sample at the center coordinate (0,0).
    It should roughly pick the value at center pixel (H//2, W//2).
    """
    B, C, H, W = 1, 2, 4, 4
    feature_map = torch.arange(B * C * H * W, dtype=torch.float32).view(B, C, H, W)

    # Sample normalized coordinate (0,0) = center
    norm_coords = torch.zeros((B, 1, 2))  # (u,v) = (0,0)

    sampled_features = bilinear_sample(feature_map, norm_coords)

    # Check output shape
    assert sampled_features.shape == (B, 1, C)

    # Check that sampled feature is close to center pixel
    center_value = feature_map[:, :, H // 2, W // 2]  # (B, C)
    assert torch.allclose(sampled_features.squeeze(1), center_value, atol=1e-2)

    print("test_bilinear_sample passed!")


if __name__ == "__main__":
    test_project_world_to_image()
    test_normalize_image_coords()
    test_bilinear_sample()
    print("All projection tests passed!")
