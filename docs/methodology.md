# SelenoFuse — Methodology

## Overview

SelenoFuse implements a multi-modal, sun-angle and scale-invariant image
correspondence pipeline for Chandrayaan-2 optical imagery. The system registers
images from OHRC (0.25 m), TMC-2 (5 m), and LRO NAC (0.5 m) sensors across
varying illumination conditions.

## Pipeline Architecture

```
Input Images → Preprocessing → Feature Detection → Matching → Geometry → Refinement → Evaluation
```

### Phase 1: Baseline (SIFT + RANSAC)

The baseline pipeline uses classical computer vision:

1. **Preprocessing**: Grayscale conversion + CLAHE contrast enhancement
2. **Feature Detection**: SIFT keypoints and 128-D descriptors
3. **Matching**: Brute-force KNN + Lowe's ratio test (threshold 0.75)
4. **Geometry**: Homography estimation via RANSAC
5. **Registration**: Perspective warp (`cv2.warpPerspective`)

### Phase 2: Illumination & Robustness

Lunar images exhibit extreme illumination variation due to the ~28-day
solar cycle and lack of atmosphere. Phase 2 addresses this with:

- **Difference of Gaussian (DoG)**: Extracts structural features by
  suppressing illumination gradients
- **Morphological Shadow Removal**: Top-hat transform to remove cast shadows
- **Homomorphic Filtering**: Log-domain frequency separation of illumination
  and reflectance
- **Single/Multi-Scale Retinex (SSR/MSR)**: Normalization in log-space to
  compress illumination dynamic range

Additional robustness comes from:
- **Cross-check matching**: Bidirectional verification of correspondences
- **Symmetric matching**: Ratio test in both directions
- **Spatial distribution**: Grid-based match selection for uniform coverage
- **MAGSAC++ / LO-RANSAC**: More robust geometric estimation

### Phase 3: Learned Features

CNN-based features trained on large datasets generalize better to the
challenging lunar surface:

- **SuperPoint**: Trained detector + descriptor with strong repeatability
  under illumination change
- **LoFTR**: Detector-free dense matcher using transformers — handles
  textureless regions where traditional detectors fail

### Phase 4: Sub-pixel Refinement

After initial homography estimation:

- **Template matching**: Local template search in warped patches
- **Iteratively Reweighted Least Squares (IRLS)**: Robust refinement
  downweighting outliers
- **Phase correlation**: Frequency-domain sub-pixel translation estimation

### Phase 5: Comprehensive Evaluation

Metrics tracked:
- **RMSE** (pixels): Reprojection error of inlier correspondences
- **Inlier ratio**: Fraction of matches surviving RANSAC
- **Spatial coverage**: Grid-cell coverage of inlier matches
- **Processing time**: Per-stage timing breakdown

## Theoretical Background

### Illumination Invariance

The image formation model for lunar surfaces is:

```
I(x, y) = L(x, y) × R(x, y) + N(x, y)
```

Where:
- `I` = observed intensity
- `L` = illumination component (slowly varying)
- `R` = reflectance (surface albedo/texture)
- `N` = sensor noise

Retinex-based methods estimate `R` by dividing out `L` in log-space:

```
log(R) = log(I) - log(L̂)
```

where `L̂` is estimated via Gaussian smoothing.

### Homography Model

For planar scenes (approximately true for lunar surface at orbital scale),
the relationship between two views is:

```
x' = H × x
```

where `H` is a 3×3 homography matrix with 8 DOF (up to scale).

## Configuration

All pipeline parameters are configurable via YAML files in `configs/`.
The system uses a layered configuration:

1. `default.yaml` — base configuration
2. Method-specific overrides (e.g., `sift.yaml`, `superpoint.yaml`)
3. Runtime overrides via CLI or Python API
