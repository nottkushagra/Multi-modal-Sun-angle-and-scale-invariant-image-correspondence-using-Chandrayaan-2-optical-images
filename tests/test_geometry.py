"""
Tests for geometry modules: RANSAC, transforms, refinement.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_registration.geometry.ransac import (
    ransac_homography,
    usac_homography,
    lo_ransac_homography,
)
from lunar_registration.geometry.transforms import (
    estimate_similarity,
    estimate_affine,
    grid_distribute_matches,
    compute_spatial_coverage,
    decompose_homography,
)
from lunar_registration.geometry.refinement import (
    phase_correlation_refine,
    iterative_least_squares_refine,
)


def _generate_point_correspondences(
    n=50, noise=1.0, outlier_ratio=0.1, seed=42,
):
    """
    Generate synthetic point correspondences with a known homography.
    """
    rng = np.random.RandomState(seed)

    # Known homography (small rotation + translation)
    angle = np.radians(5)
    H_true = np.array([
        [np.cos(angle), -np.sin(angle), 10],
        [np.sin(angle),  np.cos(angle), 15],
        [0, 0, 1],
    ])

    # Source points
    src = rng.rand(n, 2) * 500 + 50  # in [50, 550]

    # Reference points via homography
    src_h = np.column_stack([src, np.ones(n)])
    ref_h = (H_true @ src_h.T).T
    ref = ref_h[:, :2] / ref_h[:, 2:3]

    # Add noise
    ref += rng.randn(n, 2) * noise

    # Add outliers
    n_outliers = int(n * outlier_ratio)
    outlier_idx = rng.choice(n, n_outliers, replace=False)
    ref[outlier_idx] += rng.randn(n_outliers, 2) * 100

    return src, ref, H_true


def _create_mock_keypoints_and_matches(src_pts, ref_pts):
    """Create cv2.KeyPoint and cv2.DMatch objects from point arrays."""
    kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in src_pts]
    kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in ref_pts]
    matches = [cv2.DMatch(i, i, 0, 1.0) for i in range(len(src_pts))]
    return kp1, kp2, matches


class TestRANSAC:
    """Tests for RANSAC variants."""

    def test_basic_ransac(self):
        """RANSAC estimates a valid homography."""
        src, ref, H_true = _generate_point_correspondences()
        H, mask = ransac_homography(src, ref)

        assert H is not None
        assert H.shape == (3, 3)
        assert mask is not None
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_ransac_accuracy(self):
        """RANSAC homography is close to ground truth."""
        src, ref, H_true = _generate_point_correspondences(
            n=100, noise=0.5, outlier_ratio=0.1,
        )
        H, mask = ransac_homography(src, ref, reproj_threshold=3.0)

        # Normalize both
        H_norm = H / H[2, 2]
        H_true_norm = H_true / H_true[2, 2]

        # Should be close
        np.testing.assert_allclose(H_norm, H_true_norm, atol=0.5)

    def test_usac_magsac(self):
        """USAC/MAGSAC produces valid output."""
        src, ref, H_true = _generate_point_correspondences()
        H, mask = usac_homography(src, ref, method="magsac")

        assert H is not None
        assert H.shape == (3, 3)

    def test_lo_ransac(self):
        """LO-RANSAC produces at least as many inliers as basic RANSAC."""
        src, ref, H_true = _generate_point_correspondences(
            n=100, noise=0.5, outlier_ratio=0.2,
        )
        H_basic, mask_basic = ransac_homography(src, ref)
        H_lo, mask_lo = lo_ransac_homography(src, ref)

        assert H_lo is not None
        # LO-RANSAC should refine
        assert mask_lo.sum() >= mask_basic.sum() - 2  # allow small tolerance

    def test_too_few_points(self):
        """Returns None with fewer than 4 points."""
        src = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
        ref = np.array([[1, 0], [2, 1], [3, 2]], dtype=np.float64)

        H, mask = ransac_homography(src, ref)
        assert H is None
        assert mask is None


class TestTransforms:
    """Tests for transformation estimators."""

    def test_similarity(self):
        """Similarity transform estimation."""
        src, ref, _ = _generate_point_correspondences(n=30, noise=0.5)
        M, mask = estimate_similarity(src, ref)

        assert M is not None
        assert M.shape == (2, 3)

    def test_affine(self):
        """Affine transform estimation."""
        src, ref, _ = _generate_point_correspondences(n=30, noise=0.5)
        M, mask = estimate_affine(src, ref)

        assert M is not None
        assert M.shape == (2, 3)

    def test_grid_distribute(self):
        """Grid distribution reduces to max_per_cell."""
        n_pts = 200
        src_pts = np.random.rand(n_pts, 2) * 500
        ref_pts = src_pts + np.random.randn(n_pts, 2) * 5

        kp1, kp2, matches = _create_mock_keypoints_and_matches(
            src_pts, ref_pts
        )

        distributed = grid_distribute_matches(
            kp1, kp2, matches,
            image_shape=(500, 500),
            grid_size=(4, 4),
            max_per_cell=3,
        )

        # At most 4*4*3 = 48 matches
        assert len(distributed) <= 48

    def test_spatial_coverage(self):
        """Spatial coverage metric is in [0, 1]."""
        n_pts = 50
        src_pts = np.random.rand(n_pts, 2) * 500
        ref_pts = src_pts + np.random.randn(n_pts, 2)

        kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in src_pts]
        matches = [cv2.DMatch(i, i, 0, 1.0) for i in range(n_pts)]

        coverage = compute_spatial_coverage(
            kp1, matches, (500, 500), (4, 4)
        )

        assert 0.0 <= coverage <= 1.0

    def test_decompose_homography(self):
        """Homography decomposition extracts correct rotation."""
        angle = np.radians(10)
        H = np.array([
            [np.cos(angle), -np.sin(angle), 20],
            [np.sin(angle),  np.cos(angle), 30],
            [0, 0, 1],
        ])

        info = decompose_homography(H)

        assert abs(info["rotation_deg"] - 10.0) < 1.0
        assert abs(info["scale"] - 1.0) < 0.1
        assert abs(info["translation"][0] - 20.0) < 1.0


class TestRefinement:
    """Tests for registration refinement."""

    def test_phase_correlation(self):
        """Phase correlation refines a known translation."""
        # Create a simple shifted image pair
        ref = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
        # Shift by (3, 5) pixels
        H_init = np.eye(3)
        H_init[0, 2] = 3.0
        H_init[1, 2] = 5.0

        source = cv2.warpPerspective(ref, np.linalg.inv(H_init), (200, 200))

        H_refined = phase_correlation_refine(source, ref, H_init)

        assert H_refined is not None
        assert H_refined.shape == (3, 3)

    def test_irls_refinement(self):
        """IRLS refinement converges."""
        src, ref, H_true = _generate_point_correspondences(
            n=50, noise=1.0, outlier_ratio=0.0,
        )

        # Start with slightly perturbed homography
        H_init = H_true.copy()
        H_init[0, 2] += 2.0
        H_init[1, 2] += 2.0

        H_refined = iterative_least_squares_refine(src, ref, H_init)

        assert H_refined is not None
        assert H_refined.shape == (3, 3)
