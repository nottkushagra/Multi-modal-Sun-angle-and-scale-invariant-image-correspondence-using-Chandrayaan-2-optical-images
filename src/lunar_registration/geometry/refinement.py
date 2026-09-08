"""
Sub-pixel refinement methods for geometric registration.

After initial homography estimation, these methods refine the
alignment to sub-pixel accuracy using template matching,
iterative least squares, and phase correlation.
"""

import cv2
import numpy as np


def template_match_refine(
    source: np.ndarray,
    reference: np.ndarray,
    homography: np.ndarray,
    patch_size: int = 64,
    search_radius: int = 5,
    n_patches: int = 25,
) -> np.ndarray:
    """
    Refine homography using template matching on warped patches.

    Selects patches from the warped source, searches for their
    best match in the reference image, and recomputes a refined
    homography from the displacements.

    Parameters
    ----------
    source : np.ndarray
        Source image (grayscale).
    reference : np.ndarray
        Reference image (grayscale).
    homography : np.ndarray
        Initial 3×3 homography.
    patch_size : int
        Size of template patches.
    search_radius : int
        Pixel search radius around expected position.
    n_patches : int
        Number of patches to sample.

    Returns
    -------
    np.ndarray
        Refined 3×3 homography.
    """
    h, w = reference.shape[:2]

    # Warp source to reference frame
    warped = cv2.warpPerspective(source, homography, (w, h))

    half = patch_size // 2
    margin = half + search_radius

    # Sample patch centers on a grid
    rows = int(np.sqrt(n_patches))
    cols = n_patches // rows

    src_points = []
    dst_points = []

    y_step = (h - 2 * margin) / max(rows - 1, 1)
    x_step = (w - 2 * margin) / max(cols - 1, 1)

    for i in range(rows):
        for j in range(cols):
            cy = int(margin + i * y_step)
            cx = int(margin + j * x_step)

            # Extract template from warped source
            template = warped[
                cy - half : cy + half,
                cx - half : cx + half,
            ]

            if template.shape[0] != patch_size or template.shape[1] != patch_size:
                continue

            # Search window in reference
            sy = max(cy - half - search_radius, 0)
            sx = max(cx - half - search_radius, 0)
            ey = min(cy + half + search_radius, h)
            ex = min(cx + half + search_radius, w)

            search_region = reference[sy:ey, sx:ex]

            if (search_region.shape[0] < patch_size or
                    search_region.shape[1] < patch_size):
                continue

            # Template matching
            result = cv2.matchTemplate(
                search_region,
                template,
                cv2.TM_CCOEFF_NORMED,
            )

            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.6:  # quality threshold
                # Compute displacement
                best_x = sx + max_loc[0] + half
                best_y = sy + max_loc[1] + half

                src_points.append([cx, cy])
                dst_points.append([best_x, best_y])

    if len(src_points) < 4:
        return homography  # can't refine

    src_pts = np.array(src_points, dtype=np.float64).reshape(-1, 1, 2)
    dst_pts = np.array(dst_points, dtype=np.float64).reshape(-1, 1, 2)

    H_delta, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    if H_delta is None:
        return homography

    # Compose: refined = delta @ initial
    return H_delta @ homography


def iterative_least_squares_refine(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    homography: np.ndarray,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-6,
) -> np.ndarray:
    """
    Refine homography using Iteratively Reweighted Least Squares (IRLS).

    Downweights outlier correspondences based on reprojection error,
    converging to a robust estimate.

    Parameters
    ----------
    source_points : np.ndarray
        Source point coordinates, shape (N, 2).
    reference_points : np.ndarray
        Reference point coordinates, shape (N, 2).
    homography : np.ndarray
        Initial 3×3 homography.
    max_iterations : int
        Maximum IRLS iterations.
    convergence_threshold : float
        Stop when parameter change is below this value.

    Returns
    -------
    np.ndarray
        Refined 3×3 homography.
    """
    H = homography.copy()
    src = source_points.reshape(-1, 1, 2).astype(np.float64)
    ref = reference_points.reshape(-1, 2).astype(np.float64)

    for _ in range(max_iterations):
        # Compute reprojection errors
        projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        errors = np.sqrt(np.sum((ref - projected) ** 2, axis=1))

        # Huber weights
        median_error = np.median(errors)
        sigma = 1.4826 * median_error  # MAD estimator
        sigma = max(sigma, 1e-6)

        weights = np.where(
            errors < sigma,
            1.0,
            sigma / errors,
        )

        # Weighted least squares homography
        w_sqrt = np.sqrt(weights)
        src_w = src.reshape(-1, 2) * w_sqrt[:, np.newaxis]
        ref_w = ref * w_sqrt[:, np.newaxis]

        H_new, _ = cv2.findHomography(
            src_w.reshape(-1, 1, 2),
            ref_w.reshape(-1, 1, 2),
            0,  # least squares (no RANSAC)
        )

        if H_new is None:
            break

        # Check convergence
        diff = np.abs(H_new - H).max()
        H = H_new

        if diff < convergence_threshold:
            break

    return H


def phase_correlation_refine(
    source: np.ndarray,
    reference: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    """
    Refine translation component using phase correlation.

    After warping the source with the initial homography,
    computes the residual sub-pixel shift using phase correlation
    in the frequency domain.

    Parameters
    ----------
    source : np.ndarray
        Source image (grayscale).
    reference : np.ndarray
        Reference image (grayscale).
    homography : np.ndarray
        Initial 3×3 homography.

    Returns
    -------
    np.ndarray
        Refined 3×3 homography with corrected translation.
    """
    h, w = reference.shape[:2]

    # Warp source
    warped = cv2.warpPerspective(
        source, homography, (w, h)
    ).astype(np.float64)
    ref = reference.astype(np.float64)

    # Phase correlation for sub-pixel shift
    shift, response = cv2.phaseCorrelate(warped, ref)

    # Apply translation correction to homography
    T = np.eye(3)
    T[0, 2] = shift[0]
    T[1, 2] = shift[1]

    return T @ homography


def refine_registration(
    source: np.ndarray,
    reference: np.ndarray,
    homography: np.ndarray,
    source_points: np.ndarray | None = None,
    reference_points: np.ndarray | None = None,
    method: str = "template",
    **kwargs,
) -> np.ndarray:
    """
    Refine an initial homography using the specified method.

    Parameters
    ----------
    source : np.ndarray
        Source image (grayscale).
    reference : np.ndarray
        Reference image (grayscale).
    homography : np.ndarray
        Initial 3×3 homography.
    source_points : np.ndarray or None
        Source point coordinates (required for IRLS).
    reference_points : np.ndarray or None
        Reference point coordinates (required for IRLS).
    method : str
        'template', 'irls', 'phase', or 'all' (sequential).

    Returns
    -------
    np.ndarray
        Refined 3×3 homography.
    """
    H = homography.copy()

    if method == "template" or method == "all":
        H = template_match_refine(source, reference, H, **kwargs)

    if method == "irls" or method == "all":
        if source_points is not None and reference_points is not None:
            H = iterative_least_squares_refine(
                source_points, reference_points, H
            )

    if method == "phase" or method == "all":
        H = phase_correlation_refine(source, reference, H)

    return H
