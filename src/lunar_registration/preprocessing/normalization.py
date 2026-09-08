"""
Image normalization methods for lunar surface imagery.

Provides intensity normalization techniques to standardize images
from different sensors (OHRC, TMC-2, NAC) and illumination
conditions before feature extraction.
"""

import cv2
import numpy as np


def zero_mean_unit_variance(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize image to zero mean and unit variance.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.

    Returns
    -------
    np.ndarray
        Normalized image (float64).
    """
    img = image.astype(np.float64)
    mean = img.mean()
    std = img.std()

    if std < 1e-10:
        return np.zeros_like(img)

    return (img - mean) / std


def min_max_normalize(
    image: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 255.0,
) -> np.ndarray:
    """
    Min-max normalization to a target range.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    target_min : float
        Minimum output value.
    target_max : float
        Maximum output value.

    Returns
    -------
    np.ndarray
        Normalized image (uint8 if target is [0, 255]).
    """
    img = image.astype(np.float64)
    img_min, img_max = img.min(), img.max()

    if img_max - img_min < 1e-10:
        return np.full_like(
            image,
            int((target_min + target_max) / 2),
            dtype=np.uint8,
        )

    normalized = (img - img_min) / (img_max - img_min)
    scaled = normalized * (target_max - target_min) + target_min

    if target_max <= 255.0:
        return scaled.astype(np.uint8)
    return scaled


def percentile_clip_normalize(
    image: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """
    Percentile-clip normalization for satellite imagery.

    Clips pixel values at the given percentiles to remove
    outliers (e.g., sensor artifacts, saturated pixels),
    then stretches to [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    low_pct : float
        Lower percentile for clipping.
    high_pct : float
        Upper percentile for clipping.

    Returns
    -------
    np.ndarray
        Normalized image (uint8).
    """
    img = image.astype(np.float64)

    low_val = np.percentile(img, low_pct)
    high_val = np.percentile(img, high_pct)

    if high_val - low_val < 1e-10:
        return image.copy()

    clipped = np.clip(img, low_val, high_val)
    normalized = ((clipped - low_val) / (high_val - low_val) * 255)

    return normalized.astype(np.uint8)


def gamma_correction(
    image: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Apply gamma correction.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    gamma : float
        Gamma value. < 1 brightens, > 1 darkens.

    Returns
    -------
    np.ndarray
        Gamma-corrected image (uint8).
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype(np.uint8)

    return cv2.LUT(image, table)


def normalize_image(
    image: np.ndarray,
    method: str = "percentile",
    **kwargs,
) -> np.ndarray:
    """
    Normalize an image using the specified method.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    method : str
        One of: 'zero_mean', 'min_max', 'percentile', 'gamma'.

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    methods = {
        "zero_mean": zero_mean_unit_variance,
        "min_max": min_max_normalize,
        "percentile": percentile_clip_normalize,
        "gamma": gamma_correction,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(methods.keys())}"
        )

    return methods[method](image, **kwargs)
