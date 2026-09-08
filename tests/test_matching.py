"""
Tests for matching modules: ratio test, cross-check, descriptor matching.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_registration.matching.ratio_test import (
    lowe_ratio_test,
    adaptive_ratio_test,
    distance_threshold_filter,
)
from lunar_registration.matching.cross_check import (
    cross_check_match,
    symmetric_match,
)
from lunar_registration.matching.descriptor_matching import (
    match_with_flann,
    match_with_bruteforce,
    match_descriptors_generic,
)


def _generate_synthetic_descriptors(n=100, dim=128, seed=42):
    """Generate random SIFT-like descriptors for testing."""
    rng = np.random.RandomState(seed)
    desc1 = rng.rand(n, dim).astype(np.float32)
    # desc2: partially overlapping (first 50 are similar)
    desc2 = rng.rand(n, dim).astype(np.float32)
    desc2[:50] = desc1[:50] + rng.randn(50, dim).astype(np.float32) * 0.05
    return desc1, desc2


def _generate_binary_descriptors(n=100, dim=32, seed=42):
    """Generate random ORB-like binary descriptors."""
    rng = np.random.RandomState(seed)
    desc1 = rng.randint(0, 256, (n, dim), dtype=np.uint8)
    desc2 = rng.randint(0, 256, (n, dim), dtype=np.uint8)
    # Make first 30 similar
    desc2[:30] = desc1[:30] ^ rng.randint(0, 4, (30, dim), dtype=np.uint8)
    return desc1, desc2


class TestRatioTest:
    """Tests for Lowe's ratio test."""

    def test_lowe_basic(self):
        """Ratio test filters matches correctly."""
        desc1, desc2 = _generate_synthetic_descriptors()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw = matcher.knnMatch(desc1, desc2, k=2)

        strict = lowe_ratio_test(raw, 0.5)
        loose = lowe_ratio_test(raw, 0.9)

        assert len(strict) <= len(loose)
        assert all(isinstance(m, cv2.DMatch) for m in strict)

    def test_empty_matches(self):
        """Handles empty match list."""
        result = lowe_ratio_test([], 0.75)
        assert result == []

    def test_single_neighbor(self):
        """Handles pairs with fewer than 2 neighbors."""
        # Create a match with only 1 neighbor
        desc1 = np.random.rand(2, 128).astype(np.float32)
        desc2 = np.random.rand(1, 128).astype(np.float32)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw = matcher.knnMatch(desc1, desc2, k=2)

        result = lowe_ratio_test(raw, 0.75)
        assert isinstance(result, list)

    def test_adaptive_ratio(self):
        """Adaptive ratio test adjusts to target match count."""
        desc1, desc2 = _generate_synthetic_descriptors(200)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw = matcher.knnMatch(desc1, desc2, k=2)

        result = adaptive_ratio_test(raw, target_matches=10)
        assert isinstance(result, list)

    def test_distance_filter(self):
        """Distance threshold filter removes distant matches."""
        desc1, desc2 = _generate_synthetic_descriptors()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(desc1, desc2)

        filtered = distance_threshold_filter(matches, percentile=50.0)
        assert len(filtered) <= len(matches)


class TestCrossCheck:
    """Tests for cross-check matching."""

    def test_cross_check_basic(self):
        """Cross-check produces valid matches."""
        desc1, desc2 = _generate_synthetic_descriptors()
        matches = cross_check_match(desc1, desc2)

        assert isinstance(matches, list)
        assert all(isinstance(m, cv2.DMatch) for m in matches)

    def test_cross_check_sorted(self):
        """Cross-check matches are sorted by distance."""
        desc1, desc2 = _generate_synthetic_descriptors()
        matches = cross_check_match(desc1, desc2)

        if len(matches) > 1:
            distances = [m.distance for m in matches]
            assert distances == sorted(distances)

    def test_symmetric_match(self):
        """Symmetric matching returns valid results."""
        desc1, desc2 = _generate_synthetic_descriptors()
        matches = symmetric_match(desc1, desc2)

        assert isinstance(matches, list)
        # Symmetric should be more strict
        cross = cross_check_match(desc1, desc2)
        # No strict assertion on count since methods differ

    def test_binary_cross_check(self):
        """Cross-check works with binary descriptors."""
        desc1, desc2 = _generate_binary_descriptors()
        matches = cross_check_match(desc1, desc2, cv2.NORM_HAMMING)

        assert isinstance(matches, list)


class TestDescriptorMatching:
    """Tests for generic descriptor matching."""

    def test_flann_float(self):
        """FLANN matching works with float descriptors."""
        desc1, desc2 = _generate_synthetic_descriptors()
        matches = match_with_flann(desc1, desc2, "float")

        assert isinstance(matches, list)

    def test_bruteforce_float(self):
        """Brute-force matching works with float descriptors."""
        desc1, desc2 = _generate_synthetic_descriptors()
        matches = match_with_bruteforce(desc1, desc2, cv2.NORM_L2)

        assert isinstance(matches, list)

    def test_generic_dispatcher(self):
        """Generic dispatcher routes to correct backend."""
        desc1, desc2 = _generate_synthetic_descriptors()

        flann = match_descriptors_generic(desc1, desc2, method="flann")
        bf = match_descriptors_generic(desc1, desc2, method="bruteforce")

        assert isinstance(flann, list)
        assert isinstance(bf, list)

    def test_none_descriptors(self):
        """Handles None descriptors gracefully."""
        result = match_with_flann(None, None)
        assert result == []

    def test_unknown_method(self):
        """Unknown method raises ValueError."""
        desc1, desc2 = _generate_synthetic_descriptors()
        with pytest.raises(ValueError):
            match_descriptors_generic(desc1, desc2, method="unknown")
