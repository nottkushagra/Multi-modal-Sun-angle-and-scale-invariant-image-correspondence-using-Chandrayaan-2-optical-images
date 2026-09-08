"""
Ratio test implementations for feature matching.

Provides standard and adaptive versions of Lowe's ratio test
for filtering spurious feature matches.
"""

import cv2
import numpy as np


def lowe_ratio_test(
    raw_matches: list,
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    """
    Apply Lowe's ratio test to KNN matches.

    Rejects matches where the best match distance is not
    significantly better than the second-best match.

    Parameters
    ----------
    raw_matches : list
        List of KNN match pairs from cv2.BFMatcher.knnMatch.
    ratio_threshold : float
        Maximum ratio between first and second nearest neighbor
        distances. Lower values are more strict.

    Returns
    -------
    list[cv2.DMatch]
        Filtered matches that pass the ratio test.
    """
    good_matches = []

    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches


def adaptive_ratio_test(
    raw_matches: list,
    base_ratio: float = 0.75,
    min_ratio: float = 0.5,
    max_ratio: float = 0.9,
    target_matches: int = 200,
) -> list[cv2.DMatch]:
    """
    Adaptive ratio test that adjusts the threshold to reach
    a target number of matches.

    Starts with base_ratio and iteratively adjusts within
    [min_ratio, max_ratio] using binary search.

    Parameters
    ----------
    raw_matches : list
        List of KNN match pairs.
    base_ratio : float
        Initial ratio threshold.
    min_ratio : float
        Minimum allowed ratio.
    max_ratio : float
        Maximum allowed ratio.
    target_matches : int
        Desired number of matches.

    Returns
    -------
    list[cv2.DMatch]
        Filtered matches using the adapted ratio.
    """
    # First try the base ratio
    matches = lowe_ratio_test(raw_matches, base_ratio)

    if len(matches) >= target_matches:
        return matches

    # Binary search for optimal ratio
    lo, hi = base_ratio, max_ratio

    for _ in range(10):  # max 10 iterations
        mid = (lo + hi) / 2
        matches = lowe_ratio_test(raw_matches, mid)

        if len(matches) < target_matches:
            lo = mid
        else:
            hi = mid

    # Final result with the converged ratio
    result = lowe_ratio_test(raw_matches, hi)

    # Ensure we don't go below min_ratio quality
    if hi < min_ratio:
        return lowe_ratio_test(raw_matches, min_ratio)

    return result


def distance_threshold_filter(
    matches: list[cv2.DMatch],
    max_distance: float | None = None,
    percentile: float = 90.0,
) -> list[cv2.DMatch]:
    """
    Filter matches by absolute distance threshold.

    Parameters
    ----------
    matches : list[cv2.DMatch]
        Input matches.
    max_distance : float or None
        Maximum allowed distance. If None, uses percentile-based
        threshold.
    percentile : float
        Percentile of distances to use as threshold if max_distance
        is None.

    Returns
    -------
    list[cv2.DMatch]
        Filtered matches.
    """
    if not matches:
        return []

    if max_distance is None:
        distances = [m.distance for m in matches]
        max_distance = np.percentile(distances, percentile)

    return [m for m in matches if m.distance <= max_distance]
