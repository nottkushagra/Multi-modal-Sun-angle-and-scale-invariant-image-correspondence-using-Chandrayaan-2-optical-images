"""
Illumination-invariant representations for lunar surface imagery.

Implements methods to handle extreme sun-angle variations between
Chandrayaan-2 and LRO images, enabling robust feature matching
across different illumination conditions.
"""

import cv2
import numpy as np


def difference_of_gaussian(
    image: np.ndarray,
    sigma1: float = 1.0,
    sigma2: float = 3.0,
) -> np.ndarray:
    """
    Compute Difference of Gaussian (DoG) representation.

    Extracts structural features by subtracting two Gaussian-blurred
    versions of the image, suppressing illumination gradients while
    preserving edges and texture.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (uint8 or float64).
    sigma1 : float
        Standard deviation of the narrow Gaussian.
    sigma2 : float
        Standard deviation of the wide Gaussian.

    Returns
    -------
    np.ndarray
        DoG image normalized to uint8 range.
    """
    ksize1 = int(np.ceil(sigma1 * 6)) | 1  # ensure odd
    ksize2 = int(np.ceil(sigma2 * 6)) | 1

    blur1 = cv2.GaussianBlur(image, (ksize1, ksize1), sigma1)
    blur2 = cv2.GaussianBlur(image, (ksize2, ksize2), sigma2)

    dog = blur1.astype(np.float64) - blur2.astype(np.float64)

    # Normalize to [0, 255]
    dog_min, dog_max = dog.min(), dog.max()
    if dog_max - dog_min > 0:
        dog = ((dog - dog_min) / (dog_max - dog_min) * 255).astype(np.uint8)
    else:
        dog = np.zeros_like(image, dtype=np.uint8)

    return dog


def morphological_shadow_removal(
    image: np.ndarray,
    kernel_size: int = 15,
) -> np.ndarray:
    """
    Remove cast shadows using morphological top-hat transform.

    The white top-hat isolates bright features on a dark background,
    effectively removing large-scale shadow gradients common in
    low sun-angle lunar images.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    kernel_size : int
        Size of the structuring element.

    Returns
    -------
    np.ndarray
        Shadow-compensated image (uint8).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    # White top-hat: isolates bright structures
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    # Black top-hat: isolates dark structures
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    # Combine: enhance both bright and dark structures
    result = cv2.add(image, tophat)
    result = cv2.subtract(result, blackhat)

    return result


def homomorphic_filter(
    image: np.ndarray,
    gamma_low: float = 0.3,
    gamma_high: float = 2.0,
    cutoff: float = 30.0,
    order: int = 2,
) -> np.ndarray:
    """
    Apply homomorphic filtering to separate illumination from reflectance.

    Works in the log-frequency domain to compress the dynamic range
    of illumination while enhancing reflectance (surface texture).

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    gamma_low : float
        Gain for low frequencies (illumination suppression).
    gamma_high : float
        Gain for high frequencies (reflectance enhancement).
    cutoff : float
        Cutoff frequency of the Butterworth filter.
    order : int
        Order of the Butterworth filter.

    Returns
    -------
    np.ndarray
        Filtered image with reduced illumination variation (uint8).
    """
    img_float = image.astype(np.float64)
    img_float = np.where(img_float == 0, 1.0, img_float)

    # Log transform
    img_log = np.log(img_float)

    # DFT
    rows, cols = image.shape[:2]
    # Optimal DFT size
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(cols)
    padded = np.zeros((m, n), dtype=np.float64)
    padded[:rows, :cols] = img_log

    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft, axes=(0, 1))

    # Butterworth high-pass filter
    center_row, center_col = m // 2, n // 2
    u = np.arange(m).reshape(-1, 1) - center_row
    v = np.arange(n).reshape(1, -1) - center_col
    dist = np.sqrt(u ** 2 + v ** 2)

    # Avoid division by zero
    dist = np.where(dist == 0, 1e-10, dist)

    h = 1.0 / (1.0 + (cutoff / dist) ** (2 * order))
    h_filter = (gamma_high - gamma_low) * h + gamma_low
    h_filter = h_filter[:, :, np.newaxis]

    # Apply filter
    filtered = dft_shifted * h_filter
    filtered = np.fft.ifftshift(filtered, axes=(0, 1))
    result = cv2.idft(filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # Exp transform
    result = np.exp(result[:rows, :cols])

    # Normalize to uint8
    result_min, result_max = result.min(), result.max()
    if result_max - result_min > 0:
        result = ((result - result_min) / (result_max - result_min) * 255)
    else:
        result = np.zeros((rows, cols), dtype=np.float64)

    return result.astype(np.uint8)


def single_scale_retinex(
    image: np.ndarray,
    sigma: float = 80.0,
) -> np.ndarray:
    """
    Single Scale Retinex (SSR) for illumination normalization.

    Estimates and removes illumination by computing the ratio of the
    original image to a Gaussian-smoothed version in log space.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    sigma : float
        Gaussian kernel standard deviation.

    Returns
    -------
    np.ndarray
        Retinex-normalized image (uint8).
    """
    img_float = image.astype(np.float64) + 1.0  # avoid log(0)

    ksize = int(np.ceil(sigma * 6)) | 1
    blur = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
    blur = np.where(blur == 0, 1.0, blur)

    # Retinex = log(image) - log(blur)
    retinex = np.log(img_float) - np.log(blur)

    # Normalize to [0, 255]
    retinex_min, retinex_max = retinex.min(), retinex.max()
    if retinex_max - retinex_min > 0:
        retinex = ((retinex - retinex_min) / (retinex_max - retinex_min) * 255)
    else:
        retinex = np.zeros_like(image, dtype=np.float64)

    return retinex.astype(np.uint8)


def multi_scale_retinex(
    image: np.ndarray,
    sigmas: list[float] | None = None,
    weights: list[float] | None = None,
) -> np.ndarray:
    """
    Multi-Scale Retinex (MSR) combining multiple SSR scales.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    sigmas : list[float]
        Gaussian kernel sigmas for each scale.
    weights : list[float]
        Weight for each scale (must sum to 1).

    Returns
    -------
    np.ndarray
        MSR-normalized image (uint8).
    """
    if sigmas is None:
        sigmas = [15.0, 80.0, 250.0]
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    img_float = image.astype(np.float64) + 1.0

    msr = np.zeros_like(img_float)
    for sigma, weight in zip(sigmas, weights):
        ksize = int(np.ceil(sigma * 6)) | 1
        blur = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
        blur = np.where(blur == 0, 1.0, blur)
        msr += weight * (np.log(img_float) - np.log(blur))

    # Normalize
    msr_min, msr_max = msr.min(), msr.max()
    if msr_max - msr_min > 0:
        msr = ((msr - msr_min) / (msr_max - msr_min) * 255)
    else:
        msr = np.zeros_like(image, dtype=np.float64)

    return msr.astype(np.uint8)


def compute_illumination_invariant(
    image: np.ndarray,
    method: str = "retinex",
    **kwargs,
) -> np.ndarray:
    """
    Compute an illumination-invariant representation of a lunar image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    method : str
        One of: 'dog', 'shadow_removal', 'homomorphic', 'retinex', 'msr'.

    Returns
    -------
    np.ndarray
        Illumination-invariant image (uint8).
    """
    methods = {
        "dog": difference_of_gaussian,
        "shadow_removal": morphological_shadow_removal,
        "homomorphic": homomorphic_filter,
        "retinex": single_scale_retinex,
        "msr": multi_scale_retinex,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(methods.keys())}"
        )

    return methods[method](image, **kwargs)
