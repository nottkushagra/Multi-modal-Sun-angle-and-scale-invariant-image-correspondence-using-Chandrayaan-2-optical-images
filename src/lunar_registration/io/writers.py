"""
Image and registration result output writers.
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """
    Save an image to disk.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    image : np.ndarray
        Image array to save.

    Returns
    -------
    Path
        Path to the saved image file.

    Raises
    ------
    ValueError
        If the image is empty or cannot be saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image is None or image.size == 0:
        raise ValueError("Cannot save empty image")

    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to write image to {path}")

    return path


def save_homography(path: str | Path, H: np.ndarray) -> Path:
    """
    Save a 3x3 homography transformation matrix to disk.

    Parameters
    ----------
    path : str or Path
        Output path (supports .npy, .csv, or .txt).
    H : np.ndarray
        3x3 transformation matrix.

    Returns
    -------
    Path
        Path to saved homography.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if H is None or H.shape != (3, 3):
        raise ValueError(f"Invalid homography matrix shape: {getattr(H, 'shape', None)}")

    if path.suffix == ".npy":
        np.save(str(path), H)
    elif path.suffix == ".csv":
        np.savetxt(str(path), H, delimiter=",", fmt="%.8e")
    else:
        np.savetxt(str(path), H, fmt="%.8e")

    return path


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> Path:
    """
    Save evaluation metrics to JSON or text format.

    Parameters
    ----------
    path : str or Path
        Output path (.json or .txt).
    metrics : dict
        Dictionary of computed metrics.

    Returns
    -------
    Path
        Path to saved metrics file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native python for JSON serialization
    clean_metrics: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            clean_metrics[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            clean_metrics[k] = int(v)
        elif isinstance(v, np.ndarray):
            clean_metrics[k] = v.tolist()
        else:
            clean_metrics[k] = v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2)

    return path
