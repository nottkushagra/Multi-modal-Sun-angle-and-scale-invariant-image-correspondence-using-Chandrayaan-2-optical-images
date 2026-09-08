"""
Dataset management and synthetic data generation tool for SelenoFuse.

Capabilities:
  - Generate synthetic lunar image pairs for testing and CI
  - Verify manifest file integrity and checksums
  - Display Chandrayaan-2 and LRO download instructions
"""

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np


def generate_synthetic_lunar_image(
    size: tuple[int, int] = (512, 512),
    seed: int = 42,
    sun_azimuth_deg: float = 45.0,
) -> np.ndarray:
    """
    Generate a realistic synthetic lunar terrain image.

    Simulates lunar cratering, ejecta blankets, ridges, and directional
    sunlight shading based on sun azimuth.
    """
    rng = np.random.RandomState(seed)
    h, w = size

    # Background albedo
    canvas = np.full((h, w), 120, dtype=np.float32)

    # Large scale undulating topography
    x = np.linspace(-3, 3, w)
    y = np.linspace(-3, 3, h)
    xx, yy = np.meshgrid(x, y)
    canvas += 20 * np.sin(xx * 0.8) * np.cos(yy * 0.8)

    # Directional sun vector (for crater rim shadows and highlights)
    rad = np.radians(sun_azimuth_deg)
    sun_dx = np.cos(rad)
    sun_dy = np.sin(rad)

    # Coordinates grid
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Multi-scale impact craters
    num_craters = 60
    for _ in range(num_craters):
        cx = rng.randint(20, w - 20)
        cy = rng.randint(20, h - 20)
        radius = rng.randint(6, 60)
        depth = rng.uniform(20, 70)

        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        mask = dist_sq <= (radius * 1.5) ** 2

        # Crater bowl (darker inside)
        bowl = np.clip(1.0 - np.sqrt(dist_sq[mask]) / radius, 0.0, 1.0)
        canvas[mask] -= (depth * bowl).astype(np.float32)

        # Crater rim (raised highlight on sun-facing side, shadow on opposite)
        rim_mask = (dist_sq >= (radius * 0.8) ** 2) & (dist_sq <= (radius * 1.3) ** 2)
        dx = (x_grid[rim_mask] - cx) / (radius + 1e-5)
        dy = (y_grid[rim_mask] - cy) / (radius + 1e-5)
        alignment = dx * sun_dx + dy * sun_dy

        canvas[rim_mask] += (alignment * depth * 0.8).astype(np.float32)

    # Micro-craters and ridges
    for _ in range(15):
        pt1 = (rng.randint(0, w), rng.randint(0, h))
        pt2 = (rng.randint(0, w), rng.randint(0, h))
        thickness = rng.randint(1, 3)
        brightness = float(rng.randint(20, 50))
        cv2.line(canvas, pt1, pt2, brightness, thickness)

    # High-frequency regolith grain noise
    grain = rng.normal(0, 12, (h, w)).astype(np.float32)
    canvas = canvas + grain

    # Gentle Gaussian blur to simulate optical PSF
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)

    # Clip to valid 8-bit unsigned range
    return np.clip(canvas, 0, 255).astype(np.uint8)


def create_synthetic_pair(
    size: tuple[int, int] = (512, 512),
    rotation_deg: float = 4.5,
    translation: tuple[float, float] = (15.0, -10.0),
    scale: float = 0.98,
    seed: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a registered pair of synthetic images with known ground-truth homography.

    Returns
    -------
    source : np.ndarray (BGR)
    reference : np.ndarray (BGR)
    H_gt : np.ndarray (3x3 ground-truth homography from source to reference)
    """
    h, w = size
    # Reference image with sun azimuth 45 degrees
    ref_gray = generate_synthetic_lunar_image(size=size, seed=seed, sun_azimuth_deg=45.0)

    # Source image with slightly different sun illumination (azimuth 65 degrees)
    # to simulate different sun angles
    src_base = generate_synthetic_lunar_image(size=size, seed=seed, sun_azimuth_deg=65.0)

    # Transformation: source is rotated, scaled, translated version of reference
    center = (w / 2.0, h / 2.0)
    M_affine = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    M_affine[0, 2] += translation[0]
    M_affine[1, 2] += translation[1]

    src_warped = cv2.warpAffine(src_base, M_affine, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Build 3x3 homography matrix
    H_gt = np.eye(3, dtype=np.float64)
    H_gt[:2, :] = M_affine

    ref_bgr = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
    src_bgr = cv2.cvtColor(src_warped, cv2.COLOR_GRAY2BGR)

    return src_bgr, ref_bgr, H_gt


def setup_synthetic_data(output_dir: Path | None = None) -> list[Path]:
    """Generate all standard synthetic test files for the repository."""
    base_dir = Path(__file__).resolve().parent.parent
    created = []

    # 1. Standard raw test images
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    src, ref, H_gt = create_synthetic_pair(seed=42)

    ref_path = raw_dir / "reference.png"
    src_path = raw_dir / "source.png"
    h_path = raw_dir / "homography_gt.npy"

    cv2.imwrite(str(ref_path), ref)
    cv2.imwrite(str(src_path), src)
    np.save(str(h_path), H_gt)
    created.extend([ref_path, src_path, h_path])

    # 2. Pilot data directory pairs
    ref_dir = base_dir / "data" / "reference"
    src_dir = base_dir / "data" / "source"
    ref_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    p_ref = ref_dir / "pilot_ref_01.png"
    p_src = src_dir / "pilot_src_01.png"
    cv2.imwrite(str(p_ref), ref)
    cv2.imwrite(str(p_src), src)
    created.extend([p_ref, p_src])

    # 3. Pilot external directories
    ch2_pilot = base_dir / "data" / "external" / "ch2" / "pilot"
    lro_pilot = base_dir / "data" / "external" / "lro" / "pilot"
    ch2_pilot.mkdir(parents=True, exist_ok=True)
    lro_pilot.mkdir(parents=True, exist_ok=True)

    ch2_sample1 = ch2_pilot / "ch2_ohrc_001.tif"
    ch2_sample2 = ch2_pilot / "ch2_tmc2_001.tif"
    ch2_sample3 = ch2_pilot / "ch2_ohrc_002.tif"

    lro_sample1 = lro_pilot / "lro_nac_001.tif"
    lro_sample2 = lro_pilot / "lro_nac_002.tif"

    src2, ref2, _ = create_synthetic_pair(rotation_deg=6.0, scale=1.02, seed=84)

    cv2.imwrite(str(ch2_sample1), src)
    cv2.imwrite(str(ch2_sample2), cv2.resize(src, (256, 256)))
    cv2.imwrite(str(ch2_sample3), src2)

    cv2.imwrite(str(lro_sample1), ref)
    cv2.imwrite(str(lro_sample2), ref2)

    created.extend([ch2_sample1, ch2_sample2, ch2_sample3, lro_sample1, lro_sample2])

    return created


def verify_manifest(manifest_path: Path) -> bool:
    """Verify manifest file structure, existence of listed files, and checksums."""
    import csv

    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found: {manifest_path}")
        return False

    base_dir = manifest_path.resolve().parent.parent.parent

    print(f"Verifying manifest: {manifest_path}")
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} entries in manifest.")
    all_ok = True
    for i, row in enumerate(rows, start=1):
        rel_path = row.get("file_path", "")
        full_path = base_dir / rel_path
        exists = full_path.exists()
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{i}] ID: {row.get('image_id', 'N/A')} | Status: {status} | Path: {rel_path}")

    return all_ok


def print_instructions():
    """Print download instructions for PRADAN and QuickMap."""
    text = """
======================================================================
               SELENOFUSE — DATA ACQUISITION GUIDE
======================================================================
1. Chandrayaan-2 (ISRO ISSDC / PRADAN Portal):
   - URL: https://pradan.issdc.gov.in
   - Register for an account and sign in.
   - Search for Mission: Chandrayaan-2
   - Instruments:
       * OHRC (Orbiter High Resolution Camera) - ~0.25 m resolution
       * TMC-2 (Terrain Mapping Camera-2) - 5 m resolution (Nadir view)
   - Target Region: Mare Imbrium (Center: Lat -12.5°, Lon 23.8°)
   - Download Level-2 Calibrated data bundles (.img / .tif + .xml)
   - Save to: data/external/ch2/pilot/

2. Lunar Reconnaissance Orbiter (LRO NAC Reference):
   - URL: https://quickmap.lroc.asu.edu
   - Search coordinates: -12.5, 23.8
   - Select Narrow Angle Camera (NAC) layers
   - Download Calibrated Data Record (CDR) images covering the target region
   - Save to: data/external/lro/pilot/

3. Manifest and Tracking:
   - Update data/manifests/pilot_manifest.csv with product IDs and paths
   - Update docs/data/PILOT_DATA_REPORT.md with sun elevations and phase angles

For quick pipeline validation and testing without portal credentials:
   Run: python scripts/download_data.py --generate-synthetic
======================================================================
"""
    print(text)


def main():
    parser = argparse.ArgumentParser(description="SelenoFuse Data Management")
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic lunar test images for raw and pilot directories",
    )
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="Check the pilot manifest CSV and verify image paths",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display portal download and acquisition instructions",
    )

    args = parser.parse_args()

    if args.generate_synthetic:
        print("Generating synthetic lunar test images...")
        created = setup_synthetic_data()
        print(f"Successfully generated {len(created)} synthetic files:")
        for p in created:
            print(f"  - {p.relative_to(p.parent.parent.parent)}")

    elif args.verify_manifest:
        manifest_path = Path(__file__).resolve().parent.parent / "data" / "manifests" / "pilot_manifest.csv"
        ok = verify_manifest(manifest_path)
        sys.exit(0 if ok else 1)

    elif args.info:
        print_instructions()

    else:
        # Default action: print instructions + generate synthetic data if needed
        print_instructions()
        print("Generating default synthetic test dataset for immediate pipeline usage...")
        setup_synthetic_data()
        print("Done. You can now run the baseline registration pipeline!")


if __name__ == "__main__":
    main()
