"""
SuperPoint feature detector and descriptor.

Wraps the SuperPoint CNN-based feature detector/descriptor,
which is trained for robust keypoint detection under varying
illumination and viewpoint. Uses Kornia's pretrained model.

Requires: torch, kornia
"""

import numpy as np

try:
    import torch
    import kornia
    from kornia.feature import LAFDescriptor, KeyNetDetector

    SUPERPOINT_AVAILABLE = True
except ImportError:
    SUPERPOINT_AVAILABLE = False


def check_superpoint_available() -> bool:
    """Check if SuperPoint dependencies are installed."""
    return SUPERPOINT_AVAILABLE


def _image_to_tensor(image: np.ndarray) -> "torch.Tensor":
    """Convert a grayscale numpy image to a torch tensor."""
    if image.ndim == 2:
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        raise ValueError("Expected grayscale image (2D array)")
    return tensor


def detect_and_compute_superpoint(
    image: np.ndarray,
    n_features: int = 2048,
    detection_threshold: float = 0.015,
    nms_radius: int = 4,
    device: str = "cpu",
) -> tuple[list, np.ndarray | None]:
    """
    Detect SuperPoint keypoints and compute descriptors.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).
    n_features : int
        Maximum number of features.
    detection_threshold : float
        Keypoint detection confidence threshold.
    nms_radius : int
        Non-maximum suppression radius.
    device : str
        Torch device: 'cpu' or 'cuda'.

    Returns
    -------
    keypoints : list[cv2.KeyPoint]
        Detected keypoints as cv2.KeyPoint objects.
    descriptors : np.ndarray or None
        Float descriptors, shape (N, 256).

    Raises
    ------
    RuntimeError
        If torch/kornia are not installed.
    """
    if not SUPERPOINT_AVAILABLE:
        raise RuntimeError(
            "SuperPoint requires 'torch' and 'kornia'. "
            "Install with: pip install torch kornia"
        )

    import cv2 as cv2_local

    tensor = _image_to_tensor(image).to(device)

    # Use Kornia's SuperPoint
    from kornia.feature import SuperPoint as KorniaSuperPoint

    sp = KorniaSuperPoint(
        num_features=n_features,
        detection_threshold=detection_threshold,
        nms_radius=nms_radius,
    ).to(device)

    with torch.no_grad():
        output = sp.forward(tensor)

    # Extract keypoints and descriptors
    kpts = output["keypoints"][0].cpu().numpy()  # (N, 2)
    descs = output["descriptors"][0].cpu().numpy()  # (N, 256)
    scores = output["keypoint_scores"][0].cpu().numpy()  # (N,)

    # Convert to cv2.KeyPoint
    keypoints = []
    for i in range(len(kpts)):
        kp = cv2_local.KeyPoint(
            x=float(kpts[i, 0]),
            y=float(kpts[i, 1]),
            size=8.0,
            response=float(scores[i]),
        )
        keypoints.append(kp)

    return keypoints, descs.astype(np.float32)


def match_superpoint_descriptors(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    ratio_threshold: float = 0.8,
) -> list:
    """
    Match SuperPoint descriptors using L2 distance.

    Parameters
    ----------
    descriptors1 : np.ndarray
        Source descriptors, shape (N, 256).
    descriptors2 : np.ndarray
        Reference descriptors, shape (M, 256).
    ratio_threshold : float
        Lowe's ratio test threshold.

    Returns
    -------
    list[cv2.DMatch]
        Filtered matches.
    """
    import cv2 as cv2_local

    if descriptors1 is None or descriptors2 is None:
        return []

    matcher = cv2_local.BFMatcher(cv2_local.NORM_L2, crossCheck=False)
    raw = matcher.knnMatch(
        descriptors1.astype(np.float32),
        descriptors2.astype(np.float32),
        k=2,
    )

    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    return good
