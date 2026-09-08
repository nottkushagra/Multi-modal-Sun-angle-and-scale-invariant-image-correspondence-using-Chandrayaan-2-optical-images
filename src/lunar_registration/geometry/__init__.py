"""
Geometric estimation, RANSAC variations, spatial distribution, and refinement.
"""

from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.geometry.ransac import (
    estimate_homography_from_matches,
    estimate_homography_ransac,
    estimate_homography_usac,
    estimate_homography_loransac,
)
from lunar_registration.geometry.transforms import (
    estimate_similarity,
    estimate_affine,
    grid_distribute_matches,
    compute_spatial_coverage,
    decompose_homography,
)
from lunar_registration.geometry.refinement import (
    refine_registration,
    phase_correlation_refinement,
    irls_refinement,
    template_matching_refinement,
)

__all__ = [
    "estimate_homography",
    "estimate_homography_from_matches",
    "estimate_homography_ransac",
    "estimate_homography_usac",
    "estimate_homography_loransac",
    "estimate_similarity",
    "estimate_affine",
    "grid_distribute_matches",
    "compute_spatial_coverage",
    "decompose_homography",
    "refine_registration",
    "phase_correlation_refinement",
    "irls_refinement",
    "template_matching_refinement",
]
