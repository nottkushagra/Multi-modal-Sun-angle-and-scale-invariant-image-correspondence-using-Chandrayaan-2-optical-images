"""
SelenoFuse — Main entry point.

Configurable pipeline that combines preprocessing, feature
detection, matching, geometric estimation, refinement, and
evaluation into a single callable.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import load_config, get_config_value
from lunar_registration.io.loaders import load_image
from lunar_registration.preprocessing.preprocessing import preprocess
from lunar_registration.preprocessing.illumination import (
    compute_illumination_invariant,
)
from lunar_registration.preprocessing.enhancement import enhance_image
from lunar_registration.preprocessing.normalization import normalize_image
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.features.orb import (
    detect_and_compute_orb,
    match_orb_descriptors,
)
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.matching.descriptor_matching import (
    match_descriptors_generic,
)
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.geometry.ransac import (
    estimate_homography_from_matches,
)
from lunar_registration.geometry.transforms import (
    grid_distribute_matches,
    decompose_homography,
)
from lunar_registration.geometry.refinement import refine_registration
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.metrics import (
    calculate_rmse,
    calculate_inlier_ratio,
    compute_spatial_coverage_metric,
)
from lunar_registration.utils.logging import setup_logging, get_logger


def run_pipeline(
    source_path: str,
    reference_path: str,
    config_path: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """
    Run the full SelenoFuse registration pipeline.

    Parameters
    ----------
    source_path : str
        Path to the source image.
    reference_path : str
        Path to the reference image.
    config_path : str or None
        Path to config YAML (or shorthand: 'sift', 'superpoint', 'loftr').
    overrides : dict or None
        Runtime configuration overrides.

    Returns
    -------
    dict
        Pipeline results including metrics, homography, and timing.
    """
    # Load config
    cfg = load_config(config_path, overrides)
    log_level = get_config_value(cfg, "logging.level", "INFO")
    logger = setup_logging(level=log_level)

    logger.info("SelenoFuse Pipeline Starting")
    logger.info(f"Source: {source_path}")
    logger.info(f"Reference: {reference_path}")

    # --- Load images ---
    logger.info("Loading images...")
    source = load_image(source_path)
    reference = load_image(reference_path)

    # --- Preprocessing ---
    logger.info("Preprocessing...")
    source_gray = preprocess(source)
    reference_gray = preprocess(reference)

    # Illumination invariance
    if get_config_value(cfg, "preprocessing.illumination.enabled"):
        method = get_config_value(
            cfg, "preprocessing.illumination.method", "retinex"
        )
        logger.info(f"Applying illumination invariance: {method}")
        source_gray = compute_illumination_invariant(source_gray, method)
        reference_gray = compute_illumination_invariant(reference_gray, method)

    # Enhancement
    denoise = get_config_value(
        cfg, "preprocessing.enhancement.denoise", "bilateral"
    )
    if denoise != "none":
        logger.info(f"Enhancing images (denoise={denoise})...")
        source_gray = enhance_image(source_gray, denoise=denoise)
        reference_gray = enhance_image(reference_gray, denoise=denoise)

    # Normalization
    if get_config_value(cfg, "preprocessing.normalization.enabled"):
        norm_method = get_config_value(
            cfg, "preprocessing.normalization.method", "percentile"
        )
        logger.info(f"Normalizing: {norm_method}")
        source_gray = normalize_image(source_gray, norm_method)
        reference_gray = normalize_image(reference_gray, norm_method)

    # --- Feature Detection ---
    detector = get_config_value(cfg, "features.detector", "sift")
    n_features = get_config_value(cfg, "features.n_features", 5000)
    logger.info(f"Detecting features: {detector} (max={n_features})")

    if detector == "sift":
        kp1, desc1 = detect_and_compute(source_gray, n_features)
        kp2, desc2 = detect_and_compute(reference_gray, n_features)
    elif detector == "orb":
        kp1, desc1 = detect_and_compute_orb(source_gray, n_features)
        kp2, desc2 = detect_and_compute_orb(reference_gray, n_features)
    elif detector == "superpoint":
        from lunar_registration.features.superpoint import (
            detect_and_compute_superpoint,
        )
        kp1, desc1 = detect_and_compute_superpoint(source_gray, n_features)
        kp2, desc2 = detect_and_compute_superpoint(reference_gray, n_features)
    elif detector == "loftr":
        from lunar_registration.features.loftr import (
            match_loftr,
            loftr_to_cv_matches,
        )
        loftr_result = match_loftr(source_gray, reference_gray)
        kp1, kp2, matches = loftr_to_cv_matches(loftr_result)
        logger.info(f"LoFTR matches: {len(matches)}")
    else:
        raise ValueError(f"Unknown detector: {detector}")

    logger.info(f"Keypoints: source={len(kp1)}, reference={len(kp2)}")

    # --- Matching ---
    if detector != "loftr":
        match_method = get_config_value(cfg, "matching.method", "bruteforce")
        ratio = get_config_value(cfg, "matching.ratio_threshold", 0.75)
        logger.info(f"Matching: {match_method} (ratio={ratio})")

        desc_type = "binary" if detector == "orb" else "float"
        matches = match_descriptors_generic(
            desc1, desc2,
            method=match_method,
            descriptor_type=desc_type,
            ratio_threshold=ratio,
        )

    logger.info(f"Matches: {len(matches)}")

    # Spatial distribution
    if get_config_value(cfg, "matching.spatial_distribution.enabled"):
        grid = get_config_value(
            cfg, "matching.spatial_distribution.grid_size", [8, 8]
        )
        max_per = get_config_value(
            cfg, "matching.spatial_distribution.max_per_cell", 5
        )
        before = len(matches)
        matches = grid_distribute_matches(
            kp1, kp2, matches, source.shape[:2],
            tuple(grid), max_per,
        )
        logger.info(
            f"Spatial distribution: {before} → {len(matches)} matches"
        )

    if len(matches) < 4:
        logger.error("Not enough matches for homography estimation")
        return {"error": "Not enough matches", "n_matches": len(matches)}

    # --- Geometry ---
    geo_method = get_config_value(cfg, "geometry.method", "ransac")
    reproj = get_config_value(cfg, "geometry.reproj_threshold", 5.0)
    logger.info(f"Geometry: {geo_method} (threshold={reproj})")

    H, mask = estimate_homography_from_matches(
        kp1, kp2, matches, method=geo_method, reproj_threshold=reproj,
    )

    if H is None:
        logger.error("Homography estimation failed")
        return {"error": "Homography estimation failed"}

    inlier_ratio = calculate_inlier_ratio(mask)
    n_inliers = int(mask.sum())
    logger.info(f"Inliers: {n_inliers} ({inlier_ratio:.1%})")

    # Refinement
    if get_config_value(cfg, "geometry.refinement.enabled"):
        ref_method = get_config_value(
            cfg, "geometry.refinement.method", "template"
        )
        logger.info(f"Refining: {ref_method}")

        inlier_src_pts = np.float64([
            kp1[m.queryIdx].pt for m, is_in in zip(matches, mask) if is_in
        ])
        inlier_ref_pts = np.float64([
            kp2[m.trainIdx].pt for m, is_in in zip(matches, mask) if is_in
        ])

        import numpy as np
        H = refine_registration(
            source_gray, reference_gray, H,
            inlier_src_pts, inlier_ref_pts,
            method=ref_method,
        )

    # --- Registration ---
    logger.info("Warping source image...")
    registered = warp_image(source, H, reference.shape)

    # --- Evaluation ---
    inlier_matches = [
        m for m, is_in in zip(matches, mask) if is_in
    ]
    rmse = calculate_rmse(kp1, kp2, inlier_matches, H)
    coverage = compute_spatial_coverage_metric(
        kp1, inlier_matches, source.shape[:2]
    )

    # Homography decomposition
    h_info = decompose_homography(H)

    logger.info(f"RMSE: {rmse:.4f} px")
    logger.info(f"Spatial coverage: {coverage:.1%}")
    logger.info(f"Scale: {h_info['scale']:.4f}, Rotation: {h_info['rotation_deg']:.2f}°")
    logger.info("Pipeline complete!")

    return {
        "homography": H.tolist(),
        "n_keypoints_source": len(kp1),
        "n_keypoints_reference": len(kp2),
        "n_matches": len(matches),
        "n_inliers": n_inliers,
        "inlier_ratio": float(inlier_ratio),
        "rmse_pixels": float(rmse),
        "spatial_coverage": float(coverage),
        "scale": h_info["scale"],
        "rotation_deg": h_info["rotation_deg"],
        "translation": h_info["translation"],
        "registered_image": registered,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SelenoFuse Pipeline")
    parser.add_argument("--source", required=True, help="Source image path")
    parser.add_argument("--reference", required=True, help="Reference image path")
    parser.add_argument("--config", default=None, help="Config file or method name")
    args = parser.parse_args()

    result = run_pipeline(args.source, args.reference, args.config)

    if "error" not in result:
        print(f"\nRMSE: {result['rmse_pixels']:.4f} px")
        print(f"Inlier ratio: {result['inlier_ratio']:.1%}")
        print(f"Coverage: {result['spatial_coverage']:.1%}")
