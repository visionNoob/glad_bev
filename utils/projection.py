import torch
import torch.nn.functional as F


def project_world_to_image(world_coords, K, E):
    """
    Projects world (x, y) coordinates to image (u, v) coordinates.

    Args:
        world_coords: (B, Q, 2) tensor of (x, y) world coordinates
        K: (B, 3, 3) intrinsic matrices
        E: (B, 4, 4) extrinsic matrices (world to camera transform)

    Returns:
        img_coords: (B, Q, 2) tensor of (u, v) image coordinates
    """
    B, Q, _ = world_coords.shape
    device = world_coords.device

    # Append zeros for z and ones for homogeneous coordinates
    ones = torch.ones((B, Q, 1), device=device)
    zeros = torch.zeros((B, Q, 1), device=device)
    homo_world_coords = torch.cat([world_coords, zeros, ones], dim=-1)  # (B, Q, 4)

    # World to Camera
    cam_coords = torch.bmm(homo_world_coords, E.transpose(1, 2))  # (B, Q, 3)

    x, y, z = cam_coords[:, :, 0], cam_coords[:, :, 1], cam_coords[:, :, 2]

    # Project to image plane using intrinsics manually
    fx = K[:, 0, 0].unsqueeze(1)  # (B, 1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    u = fx * (x / (z + 1e-6)) + cx
    v = fy * (y / (z + 1e-6)) + cy

    img_coords = torch.stack([u, v], dim=-1)  # (B, Q, 2)
    return img_coords


def normalize_image_coords(img_coords, img_size):
    """
    Normalize (u, v) image coordinates into [-1, 1] for grid_sample.

    Args:
        img_coords: (B, Q, 2) tensor of (u, v)
        img_size: tuple (H, W)

    Returns:
        normalized_coords: (B, Q, 2)
    """
    H, W = img_size
    u, v = img_coords[:, :, 0], img_coords[:, :, 1]

    u_norm = (2.0 * u / (W - 1)) - 1.0
    v_norm = (2.0 * v / (H - 1)) - 1.0

    normalized_coords = torch.stack([u_norm, v_norm], dim=-1)
    return normalized_coords


def bilinear_sample(feature_map, norm_coords):
    """
    Bilinearly samples feature map at given normalized coordinates.

    Args:
        feature_map: (B, C, H, W) feature maps
        norm_coords: (B, Q, 2) normalized coordinates [-1, 1]

    Returns:
        sampled_features: (B, Q, C)
    """
    B, C, H, W = feature_map.shape
    Q = norm_coords.shape[1]

    # Reshape norm_coords for grid_sample: (B, Q, 1, 2)
    grid = norm_coords.unsqueeze(2)

    # Bilinear sampling
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True)

    # Reshape: (B, C, Q, 1) → (B, Q, C)
    sampled = sampled.squeeze(3).permute(0, 2, 1)
    return sampled
