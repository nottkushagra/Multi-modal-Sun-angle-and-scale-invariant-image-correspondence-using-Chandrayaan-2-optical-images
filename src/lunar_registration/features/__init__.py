"""
Feature detection and description modules (SIFT, ORB, SuperPoint, LoFTR).
"""

from lunar_registration.features.sift import detect_and_compute
from lunar_registration.features.orb import (
    detect_and_compute_orb,
    match_orb_descriptors,
)
from lunar_registration.features.superpoint import (
    detect_and_compute_superpoint,
    is_superpoint_available,
)
from lunar_registration.features.loftr import (
    match_loftr,
    is_loftr_available,
)

__all__ = [
    "detect_and_compute",
    "detect_and_compute_orb",
    "match_orb_descriptors",
    "detect_and_compute_superpoint",
    "is_superpoint_available",
    "match_loftr",
    "is_loftr_available",
]
