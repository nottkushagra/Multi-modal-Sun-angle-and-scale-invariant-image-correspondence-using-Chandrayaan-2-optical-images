"""
LoFTR (Local Feature Transformer) dense matcher.

Wraps Kornia's LoFTR implementation for detector-free,
coarse-to-fine dense matching — particularly effective
for textureless lunar surfaces where traditional keypoints
fail to detect enough features.

Requires: torch, kornia
"""

import numpy as np

try:
    import torch
    import kornia

    LOFTR_AVAILABLE = True
except ImportError:
    LOFTR_AVAILABLE = False


def check_loftr_available() -> bool:
    """Check if LoFTR dependencies are installed."""
    return LOFTR_AVAILABLE


def _image_to_tensor(image: np.ndarray) -> "torch.Tensor":
    """Convert grayscale numpy image to normalized torch tensor."""
    if image.ndim == 2:
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        raise ValueError("Expected grayscale image (2D array)")
    return tensor


def match_loftr(
    image1: np.ndarray,
    image2: np.ndarray,
    pretrained: str = "outdoor",
    confidence_threshold: float = 0.5,
    device: str = "cpu",
    max_image_size: int = 840,
) -> dict:
    """
    Dense matching using LoFTR.

    Parameters
    ----------
    image1 : np.ndarray
        Source grayscale image (uint8).
    image2 : np.ndarray
        Reference grayscale image (uint8).
    pretrained : str
        Pretrained model: 'outdoor' or 'indoor'.
    confidence_threshold : float
        Minimum match confidence to keep.
    device : str
        Torch device: 'cpu' or 'cuda'.
    max_image_size : int
        Resize images if larger (to manage memory).

    Returns
    -------
    dict
        Keys:
        - 'keypoints0': np.ndarray, shape (N, 2), source points
        - 'keypoints1': np.ndarray, shape (N, 2), reference points
        - 'confidence': np.ndarray, shape (N,), match confidences
        - 'n_matches': int
    """
    if not LOFTR_AVAILABLE:
        raise RuntimeError(
            "LoFTR requires 'torch' and 'kornia'. "
            "Install with: pip install torch kornia"
        )

    from kornia.feature import LoFTR as KorniaLoFTR

    # Resize if needed
    scale1 = 1.0
    scale2 = 1.0
    img1 = image1.copy()
    img2 = image2.copy()

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if max(h1, w1) > max_image_size:
        scale1 = max_image_size / max(h1, w1)
        import cv2
        img1 = cv2.resize(
            img1, None, fx=scale1, fy=scale1,
            interpolation=cv2.INTER_AREA,
        )

    if max(h2, w2) > max_image_size:
        scale2 = max_image_size / max(h2, w2)
        import cv2
        img2 = cv2.resize(
            img2, None, fx=scale2, fy=scale2,
            interpolation=cv2.INTER_AREA,
        )

    # Convert to tensors
    t1 = _image_to_tensor(img1).to(device)
    t2 = _image_to_tensor(img2).to(device)

    # Run LoFTR
    matcher = KorniaLoFTR(pretrained=pretrained).to(device).eval()

    with torch.no_grad():
        input_dict = {"image0": t1, "image1": t2}
        output = matcher(input_dict)

    kpts0 = output["keypoints0"].cpu().numpy()  # (N, 2)
    kpts1 = output["keypoints1"].cpu().numpy()  # (N, 2)
    confidence = output["confidence"].cpu().numpy()  # (N,)

    # Scale back to original coordinates
    if scale1 != 1.0:
        kpts0 = kpts0 / scale1
    if scale2 != 1.0:
        kpts1 = kpts1 / scale2

    # Filter by confidence
    mask = confidence >= confidence_threshold
    kpts0 = kpts0[mask]
    kpts1 = kpts1[mask]
    confidence = confidence[mask]

    return {
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "confidence": confidence,
        "n_matches": len(kpts0),
    }


def loftr_to_cv_matches(
    result: dict,
) -> tuple[list, list, list]:
    """
    Convert LoFTR output to OpenCV-compatible keypoints and matches.

    Parameters
    ----------
    result : dict
        Output from match_loftr().

    Returns
    -------
    keypoints1 : list[cv2.KeyPoint]
        Source keypoints.
    keypoints2 : list[cv2.KeyPoint]
        Reference keypoints.
    matches : list[cv2.DMatch]
        One-to-one matches (index i matches with index i).
    """
    import cv2

    keypoints1 = []
    keypoints2 = []
    matches = []

    for i in range(result["n_matches"]):
        kp1 = cv2.KeyPoint(
            x=float(result["keypoints0"][i, 0]),
            y=float(result["keypoints0"][i, 1]),
            size=8.0,
            response=float(result["confidence"][i]),
        )
        kp2 = cv2.KeyPoint(
            x=float(result["keypoints1"][i, 0]),
            y=float(result["keypoints1"][i, 1]),
            size=8.0,
            response=float(result["confidence"][i]),
        )

        keypoints1.append(kp1)
        keypoints2.append(kp2)

        match = cv2.DMatch(
            _queryIdx=i,
            _trainIdx=i,
            _distance=1.0 - float(result["confidence"][i]),
        )
        matches.append(match)

    return keypoints1, keypoints2, matches
