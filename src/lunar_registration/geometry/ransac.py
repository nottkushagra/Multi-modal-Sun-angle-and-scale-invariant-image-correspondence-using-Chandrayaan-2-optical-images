"""
Advanced RANSAC variants for robust geometric estimation.

Implements custom RANSAC with configurable parameters, LO-RANSAC
(locally optimized), and MAGSAC-style adaptive thresholding for
lunar image registration.
"""

import cv2
import numpy as np


def ransac_homography(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    reproj_threshold: float = 5.0,
    max_iterations: int = 2000,
    confidence: float = 0.999,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate homography using RANSAC with configurable parameters.

    Parameters
    ----------
    source_points : np.ndarray
        Source point coordinates, shape (N, 2) or (N, 1, 2).
    reference_points : np.ndarray
        Reference point coordinates, shape (N, 2) or (N, 1, 2).
    reproj_threshold : float
        Maximum reprojection error for inliers (pixels).
    max_iterations : int
        Maximum number of RANSAC iterations.
    confidence : float
        Confidence level (0-1) for early termination.

    Returns
    -------
    H : np.ndarray or None
        3×3 homography matrix.
    mask : np.ndarray or None
        Boolean inlier mask.
    """
    src = source_points.reshape(-1, 1, 2).astype(np.float64)
    ref = reference_points.reshape(-1, 1, 2).astype(np.float64)

    if len(src) < 4:
        return None, None

    H, mask = cv2.findHomography(
        src,
        ref,
        cv2.RANSAC,
        reproj_threshold,
        maxIters=max_iterations,
        confidence=confidence,
    )

    if H is None or mask is None:
        return None, None

    return H, mask.ravel().astype(bool)


def usac_homography(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    method: str = "magsac",
    reproj_threshold: float = 5.0,
    confidence: float = 0.999,
    max_iterations: int = 2000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate homography using OpenCV's USAC framework.

    Supports MAGSAC++, PROSAC, and other advanced estimators
    available in OpenCV 4.5+.

    Parameters
    ----------
    source_points : np.ndarray
        Source point coordinates, shape (N, 2).
    reference_points : np.ndarray
        Reference point coordinates, shape (N, 2).
    method : str
        USAC method: 'magsac', 'prosac', 'default', 'accurate', 'fast'.
    reproj_threshold : float
        Reprojection threshold (pixels).
    confidence : float
        Confidence for early termination.
    max_iterations : int
        Maximum iterations.

    Returns
    -------
    H : np.ndarray or None
        3×3 homography matrix.
    mask : np.ndarray or None
        Boolean inlier mask.
    """
    src = source_points.reshape(-1, 1, 2).astype(np.float64)
    ref = reference_points.reshape(-1, 1, 2).astype(np.float64)

    if len(src) < 4:
        return None, None

    # Map method name to OpenCV USAC flag
    usac_methods = {
        "default": cv2.USAC_DEFAULT,
        "magsac": cv2.USAC_MAGSAC,
        "prosac": cv2.USAC_PROSAC,
        "accurate": cv2.USAC_ACCURATE,
        "fast": cv2.USAC_FAST,
    }

    flag = usac_methods.get(method, cv2.USAC_MAGSAC)

    try:
        H, mask = cv2.findHomography(
            src,
            ref,
            flag,
            reproj_threshold,
            maxIters=max_iterations,
            confidence=confidence,
        )
    except cv2.error:
        # Fall back to standard RANSAC
        return ransac_homography(
            source_points, reference_points, reproj_threshold
        )

    if H is None or mask is None:
        return None, None

    return H, mask.ravel().astype(bool)


def lo_ransac_homography(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    reproj_threshold: float = 5.0,
    max_iterations: int = 2000,
    lo_iterations: int = 10,
    confidence: float = 0.999,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Locally Optimized RANSAC (LO-RANSAC) for homography estimation.

    After standard RANSAC, refines the solution by running additional
    RANSAC iterations on the inlier set, iteratively tightening the
    reprojection threshold.

    Parameters
    ----------
    source_points : np.ndarray
        Source coordinates, shape (N, 2).
    reference_points : np.ndarray
        Reference coordinates, shape (N, 2).
    reproj_threshold : float
        Initial reprojection threshold.
    max_iterations : int
        Max RANSAC iterations.
    lo_iterations : int
        Number of local optimization iterations.
    confidence : float
        Confidence for early termination.

    Returns
    -------
    H : np.ndarray or None
        Refined 3×3 homography.
    mask : np.ndarray or None
        Boolean inlier mask.
    """
    # Initial RANSAC
    H, mask = ransac_homography(
        source_points, reference_points,
        reproj_threshold, max_iterations, confidence,
    )

    if H is None or mask is None:
        return None, None

    best_inliers = int(mask.sum())
    best_H = H.copy()
    best_mask = mask.copy()

    # Local optimization: iteratively refine on inlier subset
    for i in range(lo_iterations):
        inlier_src = source_points[mask]
        inlier_ref = reference_points[mask]

        if len(inlier_src) < 4:
            break

        # Tighten threshold each iteration
        lo_threshold = reproj_threshold * (0.8 ** (i + 1))
        lo_threshold = max(lo_threshold, 1.0)

        H_lo, mask_lo = ransac_homography(
            inlier_src, inlier_ref,
            lo_threshold, max_iterations // 2,
        )

        if H_lo is None:
            continue

        # Evaluate on all points
        src_all = source_points.reshape(-1, 1, 2).astype(np.float64)
        projected = cv2.perspectiveTransform(src_all, H_lo)
        ref_all = reference_points.reshape(-1, 1, 2).astype(np.float64)
        errors = np.sqrt(
            np.sum((projected - ref_all) ** 2, axis=2)
        ).ravel()

        full_mask = errors < reproj_threshold
        n_inliers = int(full_mask.sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_H = H_lo.copy()
            best_mask = full_mask.copy()
            mask = full_mask  # update for next iteration

    return best_H, best_mask


def estimate_homography_from_matches(
    source_keypoints,
    reference_keypoints,
    matches: list,
    method: str = "ransac",
    reproj_threshold: float = 5.0,
    **kwargs,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate homography from OpenCV keypoint matches.

    Convenience function that extracts point coordinates from
    keypoints and matches, then delegates to the appropriate
    RANSAC variant.

    Parameters
    ----------
    source_keypoints : list
        Source cv2.KeyPoint objects.
    reference_keypoints : list
        Reference cv2.KeyPoint objects.
    matches : list[cv2.DMatch]
        Feature matches.
    method : str
        'ransac', 'magsac', 'lo_ransac', 'usac'.
    reproj_threshold : float
        Reprojection threshold (pixels).

    Returns
    -------
    H : np.ndarray or None
        3×3 homography matrix.
    mask : np.ndarray or None
        Boolean inlier mask.
    """
    if len(matches) < 4:
        return None, None

    src_pts = np.float64([
        source_keypoints[m.queryIdx].pt for m in matches
    ])
    ref_pts = np.float64([
        reference_keypoints[m.trainIdx].pt for m in matches
    ])

    methods = {
        "ransac": ransac_homography,
        "magsac": lambda s, r, t, **kw: usac_homography(
            s, r, "magsac", t, **kw
        ),
        "lo_ransac": lo_ransac_homography,
        "usac": lambda s, r, t, **kw: usac_homography(
            s, r, "default", t, **kw
        ),
    }

    estimator = methods.get(method, ransac_homography)
    return estimator(src_pts, ref_pts, reproj_threshold, **kwargs)
