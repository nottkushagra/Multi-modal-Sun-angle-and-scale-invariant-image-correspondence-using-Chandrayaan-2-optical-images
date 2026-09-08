"""
Benchmarking suite for comparing feature matching methods.

Runs systematic experiments across feature detectors (SIFT, ORB,
SuperPoint, LoFTR), illumination conditions, scale factors, and
cross-sensor configurations, producing structured result tables.
"""

import time
import json
from pathlib import Path

import cv2
import numpy as np

from lunar_registration.io.loaders import load_image
from lunar_registration.preprocessing.preprocessing import preprocess
from lunar_registration.preprocessing.illumination import (
    compute_illumination_invariant,
)
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.features.orb import (
    detect_and_compute_orb,
    match_orb_descriptors,
)
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.evaluation.metrics import (
    calculate_rmse,
    calculate_inlier_ratio,
    compute_spatial_coverage_metric,
    calculate_processing_time,
)


def benchmark_feature_detector(
    source_path: str,
    reference_path: str,
    detector: str = "sift",
    illumination_method: str | None = None,
    n_features: int = 5000,
) -> dict:
    """
    Benchmark a single feature detector on an image pair.

    Parameters
    ----------
    source_path : str
        Path to source image.
    reference_path : str
        Path to reference image.
    detector : str
        Feature detector: 'sift', 'orb', 'superpoint', 'loftr'.
    illumination_method : str or None
        Illumination invariance method to apply first.
    n_features : int
        Max features to detect.

    Returns
    -------
    dict
        Benchmark results including metrics and timing.
    """
    results = {
        "detector": detector,
        "illumination_method": illumination_method or "none",
        "source": str(source_path),
        "reference": str(reference_path),
    }

    t_start = time.perf_counter()

    # Load and preprocess
    source = load_image(source_path)
    reference = load_image(reference_path)
    source_gray = preprocess(source)
    reference_gray = preprocess(reference)

    # Apply illumination invariance
    if illumination_method:
        source_gray = compute_illumination_invariant(
            source_gray, illumination_method
        )
        reference_gray = compute_illumination_invariant(
            reference_gray, illumination_method
        )

    t_preprocess = time.perf_counter()

    # Feature detection and matching
    if detector == "sift":
        kp1, desc1 = detect_and_compute(source_gray, n_features)
        kp2, desc2 = detect_and_compute(reference_gray, n_features)
        t_detect = time.perf_counter()
        matches = match_descriptors(desc1, desc2)
        t_match = time.perf_counter()

    elif detector == "orb":
        kp1, desc1 = detect_and_compute_orb(source_gray, n_features)
        kp2, desc2 = detect_and_compute_orb(reference_gray, n_features)
        t_detect = time.perf_counter()
        matches = match_orb_descriptors(desc1, desc2)
        t_match = time.perf_counter()

    elif detector == "superpoint":
        from lunar_registration.features.superpoint import (
            detect_and_compute_superpoint,
            match_superpoint_descriptors,
            check_superpoint_available,
        )

        if not check_superpoint_available():
            results["error"] = "SuperPoint not available (install torch/kornia)"
            return results

        kp1, desc1 = detect_and_compute_superpoint(
            source_gray, n_features
        )
        kp2, desc2 = detect_and_compute_superpoint(
            reference_gray, n_features
        )
        t_detect = time.perf_counter()
        matches = match_superpoint_descriptors(desc1, desc2)
        t_match = time.perf_counter()

    elif detector == "loftr":
        from lunar_registration.features.loftr import (
            match_loftr,
            loftr_to_cv_matches,
            check_loftr_available,
        )

        if not check_loftr_available():
            results["error"] = "LoFTR not available (install torch/kornia)"
            return results

        t_detect = time.perf_counter()
        loftr_result = match_loftr(source_gray, reference_gray)
        kp1, kp2, matches = loftr_to_cv_matches(loftr_result)
        t_match = time.perf_counter()

    else:
        results["error"] = f"Unknown detector: {detector}"
        return results

    results["n_keypoints_source"] = len(kp1)
    results["n_keypoints_reference"] = len(kp2)
    results["n_matches"] = len(matches)

    # Homography estimation
    if len(matches) >= 4:
        try:
            H, mask = estimate_homography(kp1, kp2, matches)
            t_homography = time.perf_counter()

            inlier_ratio = calculate_inlier_ratio(mask)
            inlier_matches = [
                m for m, is_in in zip(matches, mask) if is_in
            ]
            rmse = calculate_rmse(kp1, kp2, inlier_matches, H)

            results["inlier_ratio"] = float(inlier_ratio)
            results["n_inliers"] = int(mask.sum())
            results["rmse_pixels"] = float(rmse)
            results["homography"] = H.tolist()

            # Spatial coverage
            coverage = compute_spatial_coverage_metric(
                kp1, inlier_matches, source.shape[:2]
            )
            results["spatial_coverage"] = float(coverage)

        except (RuntimeError, ValueError) as e:
            results["error"] = str(e)
            t_homography = time.perf_counter()
    else:
        results["error"] = "Not enough matches for homography"
        t_homography = time.perf_counter()

    t_end = time.perf_counter()

    # Timing
    results["time_preprocess_s"] = round(t_preprocess - t_start, 4)
    results["time_detect_s"] = round(t_detect - t_preprocess, 4)
    results["time_match_s"] = round(t_match - t_detect, 4)
    results["time_homography_s"] = round(t_homography - t_match, 4)
    results["time_total_s"] = round(t_end - t_start, 4)

    return results


def run_benchmark_suite(
    image_pairs: list[tuple[str, str]],
    detectors: list[str] | None = None,
    illumination_methods: list[str | None] | None = None,
    output_dir: str = "outputs/benchmark",
) -> list[dict]:
    """
    Run a full benchmark suite across detectors and conditions.

    Parameters
    ----------
    image_pairs : list[tuple[str, str]]
        List of (source_path, reference_path) tuples.
    detectors : list[str]
        Feature detectors to benchmark.
    illumination_methods : list[str | None]
        Illumination methods to test (None = no preprocessing).
    output_dir : str
        Directory to save results.

    Returns
    -------
    list[dict]
        All benchmark results.
    """
    if detectors is None:
        detectors = ["sift", "orb"]
    if illumination_methods is None:
        illumination_methods = [None, "retinex", "dog"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for source_path, reference_path in image_pairs:
        for detector in detectors:
            for illum in illumination_methods:
                print(
                    f"  [{detector}] + [{illum or 'none'}]: "
                    f"{Path(source_path).name} ↔ {Path(reference_path).name}"
                )

                result = benchmark_feature_detector(
                    source_path,
                    reference_path,
                    detector=detector,
                    illumination_method=illum,
                )
                all_results.append(result)

    # Save results
    results_file = output_path / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Total experiments: {len(all_results)}")

    return all_results


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table from benchmark results."""

    header = (
        f"{'Detector':<12} {'Illum':<12} "
        f"{'Matches':>8} {'Inliers':>8} "
        f"{'Ratio':>8} {'RMSE':>8} "
        f"{'Coverage':>9} {'Time(s)':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
        if "error" in r:
            print(
                f"{r['detector']:<12} {r['illumination_method']:<12} "
                f"  ERROR: {r['error']}"
            )
            continue

        print(
            f"{r['detector']:<12} {r['illumination_method']:<12} "
            f"{r.get('n_matches', 0):>8} "
            f"{r.get('n_inliers', 0):>8} "
            f"{r.get('inlier_ratio', 0):>8.3f} "
            f"{r.get('rmse_pixels', 0):>8.3f} "
            f"{r.get('spatial_coverage', 0):>9.3f} "
            f"{r.get('time_total_s', 0):>8.3f}"
        )

    print("=" * len(header))
