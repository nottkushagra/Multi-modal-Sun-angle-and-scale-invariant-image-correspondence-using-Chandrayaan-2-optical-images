"""
Preprocessing modules: normalization, enhancement, and illumination invariance.
"""

from lunar_registration.preprocessing.preprocessing import (
    to_grayscale,
    apply_clahe,
    preprocess,
)
from lunar_registration.preprocessing.illumination import (
    compute_illumination_invariant,
    difference_of_gaussians,
    single_scale_retinex,
    multi_scale_retinex,
    homomorphic_filter,
    morphological_shadow_removal,
)
from lunar_registration.preprocessing.enhancement import (
    enhance_image,
    bilateral_filter_denoise,
    unsharp_mask,
    adaptive_contrast_enhancement,
)
from lunar_registration.preprocessing.normalization import (
    normalize_image,
    min_max_normalize,
    z_score_normalize,
    percentile_clip_normalize,
)

__all__ = [
    "to_grayscale",
    "apply_clahe",
    "preprocess",
    "compute_illumination_invariant",
    "difference_of_gaussians",
    "single_scale_retinex",
    "multi_scale_retinex",
    "homomorphic_filter",
    "morphological_shadow_removal",
    "enhance_image",
    "bilateral_filter_denoise",
    "unsharp_mask",
    "adaptive_contrast_enhancement",
    "normalize_image",
    "min_max_normalize",
    "z_score_normalize",
    "percentile_clip_normalize",
]
