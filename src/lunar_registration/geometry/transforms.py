"""
Transformation models and spatial distribution analysis.

Provides similarity, affine, and homography estimators, plus
grid-based spatial match distribution to ensure uniform coverage
across the image — critical for accurate geometric estimation
of large lunar surface images.
"""

import cv2
import numpy as np


def estimate_similarity(
    source_points: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a similarity transform (rotation + scale + translation).

    Parameters
    ----------
    source_points : np.ndarray
        Source point coordinates, shape (N, 2).
    reference_points : np.ndarray
        Reference point coordinates, shape (N, 2).

    Returns
    -------
    M : np.ndarray
        2×3 similarity transform matrix.
    mask : np.ndarray
        Inlier mask.
    """
    src = source_points.reshape(-1, 1, 2).astype(np.float64)
    ref = reference_points.reshape(-1, 1, 2).astype(np.float64)

    M, mask = cv2.estimateAffinePartial2D(
        src, ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if mask is not None:
        mask = mask.ravel().astype(bool)

    return M, mask


def estimate_affine(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    reproj_threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a full affine transform (6 DOF).

    Parameters
    ----------
    source_points : np.ndarray
        Source coordinates, shape (N, 2).
    reference_points : np.ndarray
        Reference coordinates, shape (N, 2).
    reproj_threshold : float
        RANSAC inlier threshold.

    Returns
    -------
    M : np.ndarray
        2×3 affine transform matrix.
    mask : np.ndarray
        Boolean inlier mask.
    """
    src = source_points.reshape(-1, 1, 2).astype(np.float64)
    ref = reference_points.reshape(-1, 1, 2).astype(np.float64)

    M, mask = cv2.estimateAffine2D(
        src, ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
    )

    if mask is not None:
        mask = mask.ravel().astype(bool)

    return M, mask


def grid_distribute_matches(
    keypoints1,
    keypoints2,
    matches: list,
    image_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
    max_per_cell: int = 5,
) -> list:
    """
    Select spatially well-distributed matches using a grid.

    Divides the source image into a grid and selects the best
    matches per cell, ensuring uniform spatial coverage.

    Parameters
    ----------
    keypoints1 : list
        Source keypoints (cv2.KeyPoint).
    keypoints2 : list
        Reference keypoints (cv2.KeyPoint).
    matches : list[cv2.DMatch]
        Input matches.
    image_shape : tuple[int, int]
        Image dimensions as (height, width).
    grid_size : tuple[int, int]
        Number of grid rows and columns.
    max_per_cell : int
        Maximum matches to keep per grid cell.

    Returns
    -------
    list[cv2.DMatch]
        Spatially distributed matches.
    """
    h, w = image_shape[:2]
    rows, cols = grid_size
    cell_h = h / rows
    cell_w = w / cols

    # Sort matches by quality (distance)
    sorted_matches = sorted(matches, key=lambda m: m.distance)

    # Assign matches to grid cells
    grid: dict[tuple[int, int], list] = {}
    for m in sorted_matches:
        pt = keypoints1[m.queryIdx].pt
        row = min(int(pt[1] / cell_h), rows - 1)
        col = min(int(pt[0] / cell_w), cols - 1)
        cell = (row, col)

        if cell not in grid:
            grid[cell] = []

        if len(grid[cell]) < max_per_cell:
            grid[cell].append(m)

    # Flatten
    distributed = []
    for cell_matches in grid.values():
        distributed.extend(cell_matches)

    return distributed


def compute_spatial_coverage(
    keypoints,
    matches: list,
    image_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
) -> float:
    """
    Compute spatial coverage metric: fraction of grid cells
    containing at least one match.

    Parameters
    ----------
    keypoints : list
        Keypoints corresponding to queryIdx in matches.
    matches : list[cv2.DMatch]
        Feature matches.
    image_shape : tuple[int, int]
        Image dimensions (height, width).
    grid_size : tuple[int, int]
        Grid dimensions.

    Returns
    -------
    float
        Spatial coverage ratio in [0, 1].
    """
    h, w = image_shape[:2]
    rows, cols = grid_size
    cell_h = h / rows
    cell_w = w / cols

    occupied = set()
    for m in matches:
        pt = keypoints[m.queryIdx].pt
        row = min(int(pt[1] / cell_h), rows - 1)
        col = min(int(pt[0] / cell_w), cols - 1)
        occupied.add((row, col))

    return len(occupied) / (rows * cols)


def decompose_homography(
    H: np.ndarray,
) -> dict:
    """
    Decompose a homography into scale, rotation, translation,
    and perspective components.

    Parameters
    ----------
    H : np.ndarray
        3×3 homography matrix.

    Returns
    -------
    dict
        Components: 'scale', 'rotation_deg', 'translation',
        'perspective', 'det'.
    """
    # Normalize so H[2,2] = 1
    H_norm = H / H[2, 2]

    # Extract affine part
    a, b = H_norm[0, 0], H_norm[0, 1]
    c, d = H_norm[1, 0], H_norm[1, 1]

    # Scale (geometric mean of singular values)
    det = a * d - b * c
    scale = np.sqrt(abs(det))

    # Rotation
    rotation_rad = np.arctan2(c, a)
    rotation_deg = np.degrees(rotation_rad)

    # Translation
    tx, ty = H_norm[0, 2], H_norm[1, 2]

    # Perspective
    px, py = H_norm[2, 0], H_norm[2, 1]

    return {
        "scale": float(scale),
        "rotation_deg": float(rotation_deg),
        "translation": (float(tx), float(ty)),
        "perspective": (float(px), float(py)),
        "det": float(det),
    }
