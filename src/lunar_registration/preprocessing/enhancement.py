"""
Image enhancement pipeline for lunar surface imagery.

Provides multi-scale contrast enhancement, noise reduction,
and adaptive enhancement techniques tailored for the challenging
conditions of orbital lunar imaging.
"""

import cv2
import numpy as np


def multi_scale_clahe(
    image: np.ndarray,
    clip_limits: list[float] | None = None,
    tile_sizes: list[tuple[int, int]] | None = None,
    weights: list[float] | None = None,
) -> np.ndarray:
    """
    Multi-scale CLAHE combining multiple tile sizes and clip limits.

    Applies CLAHE at multiple spatial scales and blends the results
    to enhance both fine and coarse contrast features.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    clip_limits : list[float]
        CLAHE clip limits for each scale.
    tile_sizes : list[tuple[int, int]]
        Tile grid sizes for each scale.
    weights : list[float]
        Blending weights (must sum to 1).

    Returns
    -------
    np.ndarray
        Multi-scale enhanced image (uint8).
    """
    if clip_limits is None:
        clip_limits = [1.5, 2.5, 4.0]
    if tile_sizes is None:
        tile_sizes = [(4, 4), (8, 8), (16, 16)]
    if weights is None:
        weights = [1.0 / len(clip_limits)] * len(clip_limits)

    result = np.zeros(image.shape, dtype=np.float64)

    for clip_limit, tile_size, weight in zip(
        clip_limits, tile_sizes, weights
    ):
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size,
        )
        enhanced = clahe.apply(image)
        result += weight * enhanced.astype(np.float64)

    return np.clip(result, 0, 255).astype(np.uint8)


def adaptive_histogram_equalization(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid_size : tuple[int, int]
        Size of the contextual regions.

    Returns
    -------
    np.ndarray
        Equalized image (uint8).
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(image)


def bilateral_denoise(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Reduce noise while preserving edges using bilateral filtering.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).
    d : int
        Diameter of each pixel neighborhood.
    sigma_color : float
        Filter sigma in the color space.
    sigma_space : float
        Filter sigma in the coordinate space.

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def non_local_means_denoise(
    image: np.ndarray,
    h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """
    Non-local means denoising for lunar imagery.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    h : float
        Filter strength. Higher values remove more noise but
        may lose detail.
    template_window_size : int
        Size of template patch (should be odd).
    search_window_size : int
        Size of area to search for patches (should be odd).

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    return cv2.fastNlMeansDenoising(
        image,
        None,
        h,
        template_window_size,
        search_window_size,
    )


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.5,
) -> np.ndarray:
    """
    Sharpen image using unsharp masking.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    sigma : float
        Gaussian blur sigma.
    strength : float
        Sharpening strength factor.

    Returns
    -------
    np.ndarray
        Sharpened image (uint8).
    """
    ksize = int(np.ceil(sigma * 6)) | 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    sharpened = cv2.addWeighted(
        image, 1.0 + strength,
        blurred, -strength,
        0,
    )

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_image(
    image: np.ndarray,
    denoise: str = "bilateral",
    enhance_contrast: bool = True,
    sharpen: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Complete enhancement pipeline for lunar imagery.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    denoise : str
        Denoising method: 'bilateral', 'nlm', or 'none'.
    enhance_contrast : bool
        Whether to apply multi-scale CLAHE.
    sharpen : bool
        Whether to apply unsharp masking.

    Returns
    -------
    np.ndarray
        Enhanced image (uint8).
    """
    result = image.copy()

    # Step 1: Denoise
    if denoise == "bilateral":
        result = bilateral_denoise(result, **kwargs)
    elif denoise == "nlm":
        result = non_local_means_denoise(result, **kwargs)

    # Step 2: Contrast enhancement
    if enhance_contrast:
        result = multi_scale_clahe(result)

    # Step 3: Sharpen
    if sharpen:
        result = unsharp_mask(result)

    return result
