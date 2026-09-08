"""
Visualization utilities for lunar image registration.

Provides side-by-side comparisons, overlay blending, heatmaps,
match quality visualization, and spatial coverage maps.
"""

import cv2
import numpy as np


def side_by_side(
    image1: np.ndarray,
    image2: np.ndarray,
    labels: tuple[str, str] = ("Source", "Reference"),
    scale: float = 1.0,
) -> np.ndarray:
    """
    Create a side-by-side comparison image.

    Parameters
    ----------
    image1 : np.ndarray
        First image.
    image2 : np.ndarray
        Second image.
    labels : tuple[str, str]
        Labels to draw on each image.
    scale : float
        Scale factor for the output.

    Returns
    -------
    np.ndarray
        Side-by-side visualization.
    """
    # Ensure both images are BGR
    if image1.ndim == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if image2.ndim == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    # Resize to same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    target_h = max(h1, h2)

    if h1 != target_h:
        s = target_h / h1
        image1 = cv2.resize(image1, (int(w1 * s), target_h))
    if h2 != target_h:
        s = target_h / h2
        image2 = cv2.resize(image2, (int(w2 * s), target_h))

    # Concatenate
    combined = np.hstack([image1, image2])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, labels[0], (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(
        combined, labels[1],
        (image1.shape[1] + 10, 30), font, 0.8, (0, 255, 0), 2,
    )

    if scale != 1.0:
        h, w = combined.shape[:2]
        combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

    return combined


def overlay_blend(
    source: np.ndarray,
    reference: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend two images with transparency overlay.

    Parameters
    ----------
    source : np.ndarray
        Source/registered image.
    reference : np.ndarray
        Reference image.
    alpha : float
        Blending weight for source (0 = reference only).

    Returns
    -------
    np.ndarray
        Blended image.
    """
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    if reference.ndim == 2:
        reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)

    # Resize to match
    h, w = reference.shape[:2]
    source_resized = cv2.resize(source, (w, h))

    return cv2.addWeighted(source_resized, alpha, reference, 1 - alpha, 0)


def checkerboard_comparison(
    image1: np.ndarray,
    image2: np.ndarray,
    block_size: int = 50,
) -> np.ndarray:
    """
    Create a checkerboard comparison of two images.

    Alternates blocks from each image in a checkerboard pattern,
    making misalignment easy to spot.

    Parameters
    ----------
    image1 : np.ndarray
        First image.
    image2 : np.ndarray
        Second image.
    block_size : int
        Size of each checkerboard block in pixels.

    Returns
    -------
    np.ndarray
        Checkerboard comparison image.
    """
    if image1.ndim == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if image2.ndim == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    h, w = image1.shape[:2]
    image2 = cv2.resize(image2, (w, h))

    result = image1.copy()

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            # Alternate images in checkerboard pattern
            if ((y // block_size) + (x // block_size)) % 2 == 1:
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                result[y:y_end, x:x_end] = image2[y:y_end, x:x_end]

    return result


def draw_match_quality(
    source: np.ndarray,
    keypoints,
    matches,
    inlier_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Draw keypoints colored by match quality.

    Green = inlier, Red = outlier, circle size = match quality.

    Parameters
    ----------
    source : np.ndarray
        Source image.
    keypoints : list
        Source keypoints.
    matches : list[cv2.DMatch]
        Feature matches.
    inlier_mask : np.ndarray or None
        Boolean inlier mask.

    Returns
    -------
    np.ndarray
        Image with colored keypoints.
    """
    if source.ndim == 2:
        vis = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    else:
        vis = source.copy()

    for i, m in enumerate(matches):
        pt = tuple(int(c) for c in keypoints[m.queryIdx].pt)

        if inlier_mask is not None and i < len(inlier_mask):
            color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
        else:
            color = (255, 255, 0)

        # Radius inversely proportional to distance
        radius = max(3, int(15 - m.distance / 30))
        cv2.circle(vis, pt, radius, color, 1)

    return vis


def spatial_coverage_heatmap(
    keypoints,
    matches: list,
    image_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Generate a heatmap showing spatial distribution of matches.

    Parameters
    ----------
    keypoints : list
        Source keypoints.
    matches : list[cv2.DMatch]
        Feature matches.
    image_shape : tuple[int, int]
        Image dimensions (height, width).
    grid_size : tuple[int, int]
        Grid dimensions.

    Returns
    -------
    np.ndarray
        BGR heatmap image.
    """
    h, w = image_shape[:2]
    rows, cols = grid_size
    cell_h = h / rows
    cell_w = w / cols

    # Count matches per cell
    grid = np.zeros((rows, cols), dtype=np.float64)
    for m in matches:
        pt = keypoints[m.queryIdx].pt
        row = min(int(pt[1] / cell_h), rows - 1)
        col = min(int(pt[0] / cell_w), cols - 1)
        grid[row, col] += 1

    # Normalize
    max_count = grid.max()
    if max_count > 0:
        grid = grid / max_count

    # Create heatmap
    heatmap = (grid * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap
