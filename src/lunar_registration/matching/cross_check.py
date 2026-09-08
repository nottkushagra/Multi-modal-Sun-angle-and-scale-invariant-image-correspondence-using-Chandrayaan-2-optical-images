"""
Bidirectional cross-check matching for feature correspondences.

Ensures that a match (i → j) is only accepted if the reverse
match (j → i) also selects the same pair, significantly reducing
false positives.
"""

import cv2
import numpy as np


def cross_check_match(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    norm_type: int = cv2.NORM_L2,
) -> list[cv2.DMatch]:
    """
    Perform bidirectional cross-check matching.

    A match from descriptors1[i] → descriptors2[j] is accepted only
    if descriptors2[j] → descriptors1[i] is also the best match.

    Parameters
    ----------
    descriptors1 : np.ndarray
        Descriptors from the source image.
    descriptors2 : np.ndarray
        Descriptors from the reference image.
    norm_type : int
        OpenCV norm type (NORM_L2 for float, NORM_HAMMING for binary).

    Returns
    -------
    list[cv2.DMatch]
        Cross-check verified matches, sorted by distance.
    """
    matcher = cv2.BFMatcher(norm_type, crossCheck=True)

    matches = matcher.match(descriptors1, descriptors2)

    return sorted(matches, key=lambda m: m.distance)


def symmetric_match(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    norm_type: int = cv2.NORM_L2,
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    """
    Symmetric matching combining ratio test with cross-check.

    First applies Lowe's ratio test in both directions, then
    keeps only matches that are consistent in both directions.

    Parameters
    ----------
    descriptors1 : np.ndarray
        Source descriptors.
    descriptors2 : np.ndarray
        Reference descriptors.
    norm_type : int
        OpenCV norm type.
    ratio_threshold : float
        Lowe's ratio test threshold.

    Returns
    -------
    list[cv2.DMatch]
        Symmetrically verified matches.
    """
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    # Forward matching: 1 → 2
    raw_12 = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_12 = {}
    for pair in raw_12:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good_12[m.queryIdx] = m

    # Reverse matching: 2 → 1
    raw_21 = matcher.knnMatch(descriptors2, descriptors1, k=2)
    good_21 = {}
    for pair in raw_21:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good_21[m.queryIdx] = m

    # Symmetric check: match accepted if 1→2 and 2→1 agree
    symmetric_matches = []
    for idx1, match_12 in good_12.items():
        idx2 = match_12.trainIdx
        if idx2 in good_21 and good_21[idx2].trainIdx == idx1:
            symmetric_matches.append(match_12)

    return sorted(symmetric_matches, key=lambda m: m.distance)
