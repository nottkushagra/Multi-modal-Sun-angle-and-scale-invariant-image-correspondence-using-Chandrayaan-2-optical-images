"""
End-to-end registration tests using synthetic test images.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_registration.io.loaders import load_image
from lunar_registration.preprocessing.preprocessing import preprocess
from lunar_registration.preprocessing.illumination import (
    compute_illumination_invariant,
)
from lunar_registration.preprocessing.enhancement import enhance_image
from lunar_registration.preprocessing.normalization import normalize_image
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.features.orb import detect_and_compute_orb
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.metrics import (
    calculate_rmse,
    calculate_inlier_ratio,
    compute_spatial_coverage_metric,
)


def _create_synthetic_lunar_pair(
    size=(400, 400),
    shift=(10, 15),
    rotation_deg=3.0,
    seed=42,
):
    """
    Create a synthetic image pair simulating lunar surface texture
    with a known geometric transformation.
    """
    rng = np.random.RandomState(seed)
    h, w = size

    # Create base texture: simulate lunar surface
    base = np.zeros((h * 2, w * 2), dtype=np.uint8)

    # Random craters
    for _ in range(30):
        cx = rng.randint(50, w * 2 - 50)
        cy = rng.randint(50, h * 2 - 50)
        r = rng.randint(5, 40)
        brightness = rng.randint(80, 200)
        cv2.circle(base, (cx, cy), r, int(brightness), -1)

    # Random ridges
    for _ in range(10):
        x1, y1 = rng.randint(0, w * 2), rng.randint(0, h * 2)
        x2, y2 = rng.randint(0, w * 2), rng.randint(0, h * 2)
        thickness = rng.randint(1, 4)
        brightness = rng.randint(100, 180)
        cv2.line(base, (x1, y1), (x2, y2), int(brightness), thickness)

    # Add noise for texture
    noise = rng.randint(0, 30, (h * 2, w * 2), dtype=np.uint8)
    base = cv2.add(base, noise)

    # Smooth slightly
    base = cv2.GaussianBlur(base, (3, 3), 1.0)

    # Extract reference crop
    reference = base[h // 2 : h // 2 + h, w // 2 : w // 2 + w]

    # Create transformed source
    center = (w // 2, h // 2)
    angle = rotation_deg
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += shift[0]
    M[1, 2] += shift[1]

    source = cv2.warpAffine(reference, M, (w, h))

    # Convert to BGR for pipeline compatibility
    reference_bgr = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    source_bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

    return source_bgr, reference_bgr


class TestEndToEndRegistration:
    """End-to-end registration tests."""

    def test_sift_registration(self):
        """SIFT-based registration on synthetic pair."""
        source, reference = _create_synthetic_lunar_pair()

        src_gray = preprocess(source)
        ref_gray = preprocess(reference)

        kp1, desc1 = detect_and_compute(src_gray)
        kp2, desc2 = detect_and_compute(ref_gray)

        assert len(kp1) > 10, "Too few source keypoints"
        assert len(kp2) > 10, "Too few reference keypoints"

        matches = match_descriptors(desc1, desc2)
        assert len(matches) >= 4, "Too few matches"

        H, mask = estimate_homography(kp1, kp2, matches)

        assert H is not None
        assert H.shape == (3, 3)
        assert mask.sum() > 0

        # Register
        registered = warp_image(source, H, reference.shape)
        assert registered.shape == reference.shape

        # RMSE should be reasonable
        inlier_matches = [
            m for m, is_in in zip(matches, mask) if is_in
        ]
        rmse = calculate_rmse(kp1, kp2, inlier_matches, H)
        assert rmse < 10.0, f"RMSE too high: {rmse}"

    def test_orb_registration(self):
        """ORB-based registration on synthetic pair."""
        source, reference = _create_synthetic_lunar_pair()

        src_gray = preprocess(source)
        ref_gray = preprocess(reference)

        kp1, desc1 = detect_and_compute_orb(src_gray)
        kp2, desc2 = detect_and_compute_orb(ref_gray)

        assert len(kp1) > 0, "No ORB keypoints detected"

        if desc1 is not None and desc2 is not None:
            from lunar_registration.features.orb import match_orb_descriptors
            matches = match_orb_descriptors(desc1, desc2)

            if len(matches) >= 4:
                H, mask = estimate_homography(kp1, kp2, matches)
                assert H is not None

    def test_illumination_preprocessing(self):
        """Illumination invariance methods don't crash on synthetic data."""
        source, reference = _create_synthetic_lunar_pair()
        src_gray = preprocess(source)

        for method in ["dog", "retinex", "shadow_removal", "homomorphic", "msr"]:
            result = compute_illumination_invariant(src_gray, method)
            assert result.shape == src_gray.shape
            assert result.dtype == np.uint8

    def test_enhancement_pipeline(self):
        """Enhancement pipeline runs without errors."""
        source, _ = _create_synthetic_lunar_pair()
        src_gray = preprocess(source)

        enhanced = enhance_image(
            src_gray, denoise="bilateral", enhance_contrast=True
        )
        assert enhanced.shape == src_gray.shape

    def test_normalization_methods(self):
        """All normalization methods produce valid output."""
        _, reference = _create_synthetic_lunar_pair()
        ref_gray = preprocess(reference)

        for method in ["zero_mean", "min_max", "percentile", "gamma"]:
            result = normalize_image(ref_gray, method)
            assert result is not None

    def test_spatial_coverage_metric(self):
        """Spatial coverage is computed correctly."""
        source, reference = _create_synthetic_lunar_pair()
        src_gray = preprocess(source)
        ref_gray = preprocess(reference)

        kp1, desc1 = detect_and_compute(src_gray)
        kp2, desc2 = detect_and_compute(ref_gray)
        matches = match_descriptors(desc1, desc2)

        if len(matches) >= 4:
            H, mask = estimate_homography(kp1, kp2, matches)
            inlier_matches = [
                m for m, is_in in zip(matches, mask) if is_in
            ]
            coverage = compute_spatial_coverage_metric(
                kp1, inlier_matches, source.shape[:2]
            )
            assert 0.0 <= coverage <= 1.0

    def test_inlier_ratio(self):
        """Inlier ratio is in valid range."""
        mask = np.array([True, True, False, True, False])
        ratio = calculate_inlier_ratio(mask)
        assert ratio == pytest.approx(0.6)

    def test_empty_inlier_mask(self):
        """Empty mask returns 0."""
        ratio = calculate_inlier_ratio(np.array([]))
        assert ratio == 0.0


class TestSyntheticImageCreation:
    """Test the synthetic image generator itself."""

    def test_creates_valid_images(self):
        """Synthetic pair has correct shape and type."""
        source, reference = _create_synthetic_lunar_pair(size=(300, 300))

        assert source.shape == (300, 300, 3)
        assert reference.shape == (300, 300, 3)
        assert source.dtype == np.uint8
        assert reference.dtype == np.uint8

    def test_different_seeds(self):
        """Different seeds produce different images."""
        s1, _ = _create_synthetic_lunar_pair(seed=1)
        s2, _ = _create_synthetic_lunar_pair(seed=2)

        # Should not be identical
        assert not np.array_equal(s1, s2)
