import cv2
import numpy as np


def calculate_rmse(
    source_keypoints,
    reference_keypoints,
    matches,
    homography: np.ndarray,
) -> float:
    """
    Calculate Root Mean Square Error of reprojection.

    Parameters
    ----------
    source_keypoints : list
        Source cv2.KeyPoint objects.
    reference_keypoints : list
        Reference cv2.KeyPoint objects.
    matches : list[cv2.DMatch]
        Feature matches (typically inlier matches only).
    homography : np.ndarray
        3×3 homography matrix.

    Returns
    -------
    float
        RMSE in pixels.
    """
    source_points = np.float32(
        [
            source_keypoints[m.queryIdx].pt
            for m in matches
        ]
    ).reshape(-1, 1, 2)

    reference_points = np.float32(
        [
            reference_keypoints[m.trainIdx].pt
            for m in matches
        ]
    ).reshape(-1, 1, 2)

    projected_points = cv2.perspectiveTransform(
        source_points,
        homography,
    )

    errors = (
        reference_points - projected_points
    ).reshape(-1, 2)

    squared_errors = np.sum(
        errors ** 2,
        axis=1,
    )

    return float(
        np.sqrt(np.mean(squared_errors))
    )


def calculate_inlier_ratio(
    inlier_mask: np.ndarray,
) -> float:
    """
    Calculate the ratio of inliers to total matches.

    Parameters
    ----------
    inlier_mask : np.ndarray
        Boolean mask of inliers.

    Returns
    -------
    float
        Inlier ratio in [0, 1].
    """
    if len(inlier_mask) == 0:
        return 0.0

    return float(np.mean(inlier_mask))


def compute_spatial_coverage_metric(
    keypoints,
    matches: list,
    image_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
) -> float:
    """
    Compute spatial coverage: fraction of grid cells containing
    at least one inlier match.

    Parameters
    ----------
    keypoints : list
        Keypoints indexed by matches' queryIdx.
    matches : list[cv2.DMatch]
        Feature matches.
    image_shape : tuple[int, int]
        Image dimensions (height, width).
    grid_size : tuple[int, int]
        Grid dimensions (rows, cols).

    Returns
    -------
    float
        Coverage ratio in [0, 1].
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


def calculate_reprojection_errors(
    source_keypoints,
    reference_keypoints,
    matches,
    homography: np.ndarray,
) -> np.ndarray:
    """
    Compute per-match reprojection errors.

    Parameters
    ----------
    source_keypoints : list
        Source keypoints.
    reference_keypoints : list
        Reference keypoints.
    matches : list[cv2.DMatch]
        Feature matches.
    homography : np.ndarray
        3×3 homography.

    Returns
    -------
    np.ndarray
        Per-match reprojection errors in pixels.
    """
    src = np.float32(
        [source_keypoints[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    ref = np.float32(
        [reference_keypoints[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(src, homography)

    errors = np.sqrt(
        np.sum((ref - projected).reshape(-1, 2) ** 2, axis=1)
    )

    return errors


def calculate_processing_time(func, *args, **kwargs) -> tuple:
    """
    Measure execution time of a function.

    Parameters
    ----------
    func : callable
        Function to time.
    *args, **kwargs
        Arguments to pass to the function.

    Returns
    -------
    result
        Function return value.
    elapsed : float
        Elapsed time in seconds.
    """
    import time

    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start

    return result, elapsed


def compute_region_inlier_counts(
    keypoints,
    inlier_mask: np.ndarray,
    matches: list,
    image_shape: tuple[int, int],
    grid_size: tuple[int, int] = (4, 4),
) -> dict:
    """
    Count inliers per grid region.

    Parameters
    ----------
    keypoints : list
        Source keypoints.
    inlier_mask : np.ndarray
        Boolean inlier mask.
    matches : list[cv2.DMatch]
        Feature matches.
    image_shape : tuple[int, int]
        Image dimensions (height, width).
    grid_size : tuple[int, int]
        Grid dimensions (rows, cols).

    Returns
    -------
    dict
        Mapping from (row, col) to inlier count.
    """
    h, w = image_shape[:2]
    rows, cols = grid_size
    cell_h = h / rows
    cell_w = w / cols

    counts: dict[tuple[int, int], int] = {}

    for m, is_inlier in zip(matches, inlier_mask):
        if not is_inlier:
            continue

        pt = keypoints[m.queryIdx].pt
        row = min(int(pt[1] / cell_h), rows - 1)
        col = min(int(pt[0] / cell_w), cols - 1)

        cell = (row, col)
        counts[cell] = counts.get(cell, 0) + 1

    return counts