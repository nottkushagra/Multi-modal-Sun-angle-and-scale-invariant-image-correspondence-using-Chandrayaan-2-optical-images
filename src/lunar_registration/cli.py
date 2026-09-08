import argparse
import json
from pathlib import Path

import cv2

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
from lunar_registration.matching.descriptor_matching import (
    match_descriptors_generic,
)
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.geometry.transforms import grid_distribute_matches
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.metrics import (
    calculate_rmse,
    calculate_inlier_ratio,
    compute_spatial_coverage_metric,
)
from lunar_registration.evaluation.visualization import draw_matches


def register_images(source_path, reference_path, detector="sift",
                    illumination=None, ratio=0.75):
    """Run the registration pipeline with configurable detector."""

    print("Loading images...")
    source = load_image(source_path)
    reference = load_image(reference_path)

    print("Preprocessing...")
    source_processed = preprocess(source)
    reference_processed = preprocess(reference)

    # Optional illumination invariance
    if illumination:
        print(f"Applying illumination invariance: {illumination}")
        source_processed = compute_illumination_invariant(
            source_processed, illumination
        )
        reference_processed = compute_illumination_invariant(
            reference_processed, illumination
        )

    print(f"Detecting {detector.upper()} features...")

    if detector == "sift":
        kp1, desc1 = detect_and_compute(source_processed)
        kp2, desc2 = detect_and_compute(reference_processed)
        matches = match_descriptors(desc1, desc2, ratio)
    elif detector == "orb":
        kp1, desc1 = detect_and_compute_orb(source_processed)
        kp2, desc2 = detect_and_compute_orb(reference_processed)
        matches = match_orb_descriptors(desc1, desc2, ratio)
    else:
        raise ValueError(f"Unknown detector: {detector}")

    print(f"Keypoints: source={len(kp1)}, reference={len(kp2)}")
    print(f"Good matches: {len(matches)}")

    if len(matches) < 4:
        raise RuntimeError("Not enough matches for homography estimation.")

    print("Estimating homography with RANSAC...")
    homography, inlier_mask = estimate_homography(kp1, kp2, matches)

    inlier_count = int(inlier_mask.sum())
    inlier_ratio = calculate_inlier_ratio(inlier_mask)

    print(f"Inliers: {inlier_count}")
    print(f"Inlier ratio: {inlier_ratio:.3f}")

    print("Warping source image...")
    registered = warp_image(source, homography, reference.shape)

    print("Calculating metrics...")
    inlier_matches = [
        match for match, is_inlier
        in zip(matches, inlier_mask) if is_inlier
    ]

    rmse = calculate_rmse(kp1, kp2, inlier_matches, homography)
    coverage = compute_spatial_coverage_metric(
        kp1, inlier_matches, source.shape[:2]
    )

    print(f"RMSE: {rmse:.4f} pixels")
    print(f"Spatial coverage: {coverage:.3f}")

    # Save outputs
    output_dir = Path("outputs")
    matches_dir = output_dir / "matches"
    registered_dir = output_dir / "registered"
    metrics_dir = output_dir / "metrics"

    for d in [matches_dir, registered_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    match_visualization = draw_matches(
        source, kp1, reference, kp2, matches, inlier_mask,
    )

    cv2.imwrite(str(matches_dir / "matches.png"), match_visualization)
    cv2.imwrite(str(registered_dir / "registered.png"), registered)

    metrics = {
        "detector": detector,
        "illumination": illumination,
        "source_keypoints": len(kp1),
        "reference_keypoints": len(kp2),
        "good_matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "rmse_pixels": rmse,
        "spatial_coverage": coverage,
        "homography": homography.tolist(),
    }

    with open(metrics_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nRegistration complete.")
    print(f"Registered image: {registered_dir / 'registered.png'}")
    print(f"Matches: {matches_dir / 'matches.png'}")
    print(f"Metrics: {metrics_dir / 'results.json'}")


def preprocess_images(source_path, reference_path, method="retinex"):
    """Run preprocessing pipeline and save results."""
    print("Loading images...")
    source = load_image(source_path)
    reference = load_image(reference_path)

    print("Preprocessing...")
    source_gray = preprocess(source)
    reference_gray = preprocess(reference)

    print(f"Applying illumination invariance: {method}")
    source_inv = compute_illumination_invariant(source_gray, method)
    reference_inv = compute_illumination_invariant(reference_gray, method)

    output_dir = Path("outputs") / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "source_preprocessed.png"), source_inv)
    cv2.imwrite(str(output_dir / "reference_preprocessed.png"), reference_inv)

    print(f"Saved to: {output_dir}")


def run_benchmark(image_pairs_file, detectors=None, output_dir=None):
    """Run the benchmark suite."""
    from lunar_registration.evaluation.benchmark import (
        run_benchmark_suite,
        print_comparison_table,
    )

    # Load image pairs from JSON
    with open(image_pairs_file) as f:
        pairs_data = json.load(f)

    image_pairs = [
        (p["source"], p["reference"]) for p in pairs_data
    ]

    if detectors is None:
        detectors = ["sift", "orb"]

    if output_dir is None:
        output_dir = "outputs/benchmark"

    print(f"Running benchmark: {len(image_pairs)} pairs × {len(detectors)} detectors")
    results = run_benchmark_suite(image_pairs, detectors, output_dir=output_dir)

    print_comparison_table(results)


def main():
    parser = argparse.ArgumentParser(
        description="SelenoFuse — Lunar Image Registration"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Register command
    register_parser = subparsers.add_parser(
        "register",
        help="Register a source image to a reference",
    )
    register_parser.add_argument(
        "--source", required=True,
        help="Path to source image",
    )
    register_parser.add_argument(
        "--reference", required=True,
        help="Path to reference image",
    )
    register_parser.add_argument(
        "--detector", default="sift",
        choices=["sift", "orb"],
        help="Feature detector (default: sift)",
    )
    register_parser.add_argument(
        "--illumination", default=None,
        choices=["dog", "shadow_removal", "homomorphic", "retinex", "msr"],
        help="Illumination invariance method",
    )
    register_parser.add_argument(
        "--ratio", type=float, default=0.75,
        help="Match ratio threshold (default: 0.75)",
    )

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Run preprocessing pipeline",
    )
    preprocess_parser.add_argument(
        "--source", required=True,
        help="Path to source image",
    )
    preprocess_parser.add_argument(
        "--reference", required=True,
        help="Path to reference image",
    )
    preprocess_parser.add_argument(
        "--method", default="retinex",
        choices=["dog", "shadow_removal", "homomorphic", "retinex", "msr"],
        help="Illumination method (default: retinex)",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmark suite",
    )
    benchmark_parser.add_argument(
        "--pairs", required=True,
        help="JSON file with image pairs",
    )
    benchmark_parser.add_argument(
        "--detectors", nargs="+", default=["sift", "orb"],
        help="Detectors to benchmark",
    )
    benchmark_parser.add_argument(
        "--output", default="outputs/benchmark",
        help="Output directory",
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a registration result",
    )
    evaluate_parser.add_argument(
        "--source", required=True,
        help="Path to source image",
    )
    evaluate_parser.add_argument(
        "--reference", required=True,
        help="Path to reference image",
    )
    evaluate_parser.add_argument(
        "--registered", required=True,
        help="Path to registered image",
    )

    args = parser.parse_args()

    if args.command == "register":
        register_images(
            args.source, args.reference,
            detector=args.detector,
            illumination=args.illumination,
            ratio=args.ratio,
        )
    elif args.command == "preprocess":
        preprocess_images(
            args.source, args.reference,
            method=args.method,
        )
    elif args.command == "benchmark":
        run_benchmark(
            args.pairs,
            detectors=args.detectors,
            output_dir=args.output,
        )
    elif args.command == "evaluate":
        print("Evaluation mode: comparing registered vs reference")
        import numpy as np
        reg = cv2.imread(args.registered, cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
        if reg is None or ref is None:
            print("Error: could not load images")
            return
        # Resize if needed
        if reg.shape != ref.shape:
            reg = cv2.resize(reg, (ref.shape[1], ref.shape[0]))
        diff = np.abs(reg.astype(float) - ref.astype(float))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        print(f"Pixel RMSE: {rmse:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()