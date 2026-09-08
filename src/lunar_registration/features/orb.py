"""
ORB (Oriented FAST and Rotated BRIEF) feature detection.

Provides a fast, rotation-invariant binary feature detector
as an alternative to SIFT for real-time applications and
as a baseline comparator.
"""

import cv2
import numpy as np


def detect_and_compute_orb(
    image: np.ndarray,
    n_features: int = 5000,
    scale_factor: float = 1.2,
    n_levels: int = 8,
    edge_threshold: int = 31,
    patch_size: int = 31,
) -> tuple[list, np.ndarray | None]:
    """
    Detect ORB keypoints and compute binary descriptors.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    n_features : int
        Maximum number of features to detect.
    scale_factor : float
        Pyramid decimation ratio (> 1.0).
    n_levels : int
        Number of pyramid levels.
    edge_threshold : int
        Border size where features are not detected.
    patch_size : int
        Size of the patch used to compute BRIEF descriptors.

    Returns
    -------
    keypoints : list[cv2.KeyPoint]
        Detected keypoints.
    descriptors : np.ndarray or None
        Binary descriptors (uint8), shape (N, 32).
    """
    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=edge_threshold,
        patchSize=patch_size,
    )

    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def match_orb_descriptors(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    ratio_threshold: float = 0.75,
    use_cross_check: bool = True,
) -> list[cv2.DMatch]:
    """
    Match ORB binary descriptors using Hamming distance.

    Parameters
    ----------
    descriptors1 : np.ndarray
        Source ORB descriptors.
    descriptors2 : np.ndarray
        Reference ORB descriptors.
    ratio_threshold : float
        Lowe's ratio threshold (used if cross_check=False).
    use_cross_check : bool
        If True, use simpler cross-check matching.

    Returns
    -------
    list[cv2.DMatch]
        Filtered matches.
    """
    if descriptors1 is None or descriptors2 is None:
        return []

    if use_cross_check:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)
        return sorted(matches, key=lambda m: m.distance)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches
