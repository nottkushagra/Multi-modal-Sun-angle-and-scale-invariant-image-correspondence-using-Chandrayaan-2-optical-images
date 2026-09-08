"""
Feature descriptor matching modules.
"""

from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.matching.ratio_test import (
    apply_ratio_test,
    adaptive_ratio_test,
    filter_matches_by_distance,
)
from lunar_registration.matching.cross_check import (
    cross_check_matches,
    symmetric_match,
    cross_check_binary_matches,
)
from lunar_registration.matching.descriptor_matching import (
    match_descriptors_generic,
    flann_match_descriptors,
    bruteforce_match_descriptors,
)

__all__ = [
    "match_descriptors",
    "apply_ratio_test",
    "adaptive_ratio_test",
    "filter_matches_by_distance",
    "cross_check_matches",
    "symmetric_match",
    "cross_check_binary_matches",
    "match_descriptors_generic",
    "flann_match_descriptors",
    "bruteforce_match_descriptors",
]
