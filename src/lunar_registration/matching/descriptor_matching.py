"""
Generic descriptor matching with multiple backend strategies.

Supports both float (SIFT, SuperPoint) and binary (ORB) descriptors
with FLANN and brute-force backends.
"""

import cv2
import numpy as np

from lunar_registration.matching.ratio_test import lowe_ratio_test
from lunar_registration.matching.cross_check import (
    cross_check_match,
    symmetric_match,
)


def create_flann_matcher(
    descriptor_type: str = "float",
) -> cv2.FlannBasedMatcher:
    """
    Create a FLANN-based matcher for the given descriptor type.

    Parameters
    ----------
    descriptor_type : str
        'float' for SIFT/SuperPoint, 'binary' for ORB.

    Returns
    -------
    cv2.FlannBasedMatcher
        Configured FLANN matcher.
    """
    if descriptor_type == "float":
        index_params = dict(
            algorithm=1,  # FLANN_INDEX_KDTREE
            trees=5,
        )
        search_params = dict(checks=50)
    elif descriptor_type == "binary":
        index_params = dict(
            algorithm=6,  # FLANN_INDEX_LSH
            table_number=12,
            key_size=20,
            multi_probe_level=2,
        )
        search_params = dict(checks=50)
    else:
        raise ValueError(
            f"Unknown descriptor type: {descriptor_type}. "
            "Use 'float' or 'binary'."
        )

    return cv2.FlannBasedMatcher(index_params, search_params)


def match_with_flann(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    descriptor_type: str = "float",
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    """
    Match descriptors using FLANN with Lowe's ratio test.

    FLANN is significantly faster than brute-force for large
    descriptor sets (> 1000 features).

    Parameters
    ----------
    descriptors1 : np.ndarray
        Source descriptors.
    descriptors2 : np.ndarray
        Reference descriptors.
    descriptor_type : str
        'float' for SIFT/SuperPoint, 'binary' for ORB.
    ratio_threshold : float
        Lowe's ratio test threshold.

    Returns
    -------
    list[cv2.DMatch]
        Filtered matches.
    """
    if descriptors1 is None or descriptors2 is None:
        return []

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return []

    matcher = create_flann_matcher(descriptor_type)

    # Ensure correct dtype for FLANN
    if descriptor_type == "float":
        descriptors1 = descriptors1.astype(np.float32)
        descriptors2 = descriptors2.astype(np.float32)

    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    return lowe_ratio_test(raw_matches, ratio_threshold)


def match_with_bruteforce(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    norm_type: int = cv2.NORM_L2,
    ratio_threshold: float = 0.75,
    use_cross_check: bool = False,
) -> list[cv2.DMatch]:
    """
    Match descriptors using brute-force matching.

    Parameters
    ----------
    descriptors1 : np.ndarray
        Source descriptors.
    descriptors2 : np.ndarray
        Reference descriptors.
    norm_type : int
        OpenCV norm: NORM_L2 for float, NORM_HAMMING for binary.
    ratio_threshold : float
        Lowe's ratio test threshold (ignored if use_cross_check=True).
    use_cross_check : bool
        If True, use cross-check instead of ratio test.

    Returns
    -------
    list[cv2.DMatch]
        Filtered matches.
    """
    if descriptors1 is None or descriptors2 is None:
        return []

    if use_cross_check:
        return cross_check_match(descriptors1, descriptors2, norm_type)

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    return lowe_ratio_test(raw_matches, ratio_threshold)


def match_descriptors_generic(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    method: str = "flann",
    descriptor_type: str = "float",
    ratio_threshold: float = 0.75,
    symmetric: bool = False,
) -> list[cv2.DMatch]:
    """
    Generic descriptor matching dispatcher.

    Parameters
    ----------
    descriptors1 : np.ndarray
        Source descriptors.
    descriptors2 : np.ndarray
        Reference descriptors.
    method : str
        'flann', 'bruteforce', or 'symmetric'.
    descriptor_type : str
        'float' or 'binary'.
    ratio_threshold : float
        Lowe's ratio threshold.
    symmetric : bool
        If True, apply symmetric (bidirectional) matching.

    Returns
    -------
    list[cv2.DMatch]
        Filtered matches.
    """
    if symmetric or method == "symmetric":
        norm = (
            cv2.NORM_HAMMING
            if descriptor_type == "binary"
            else cv2.NORM_L2
        )
        return symmetric_match(
            descriptors1, descriptors2, norm, ratio_threshold
        )

    if method == "flann":
        return match_with_flann(
            descriptors1, descriptors2, descriptor_type, ratio_threshold
        )

    if method == "bruteforce":
        norm = (
            cv2.NORM_HAMMING
            if descriptor_type == "binary"
            else cv2.NORM_L2
        )
        return match_with_bruteforce(
            descriptors1, descriptors2, norm, ratio_threshold
        )

    raise ValueError(
        f"Unknown matching method: {method}. "
        "Use 'flann', 'bruteforce', or 'symmetric'."
    )
