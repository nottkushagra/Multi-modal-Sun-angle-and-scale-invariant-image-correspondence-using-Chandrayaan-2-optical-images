# SelenoFuse — Experiment Design

## Overview

This document outlines the experimental protocol for evaluating the
SelenoFuse pipeline across multiple dimensions: feature detector choice,
illumination preprocessing, scale invariance, and cross-sensor matching.

## Experiment Matrix

### E1: Feature Detector Comparison

| Detector | Type | Descriptor Dim | Invariance |
|----------|------|---------------|------------|
| SIFT | Handcrafted | 128 (float) | Scale, Rotation |
| ORB | Handcrafted | 32 (binary) | Scale, Rotation |
| SuperPoint | Learned (CNN) | 256 (float) | Illumination, Viewpoint |
| LoFTR | Learned (Transformer) | Dense | Textureless, Illumination |

**Protocol**:
1. Fix illumination preprocessing to "none"
2. Run each detector on all image pairs
3. Record: n_keypoints, n_matches, inlier_ratio, RMSE, coverage, time

### E2: Illumination Robustness

| Method | Type | Target |
|--------|------|--------|
| None | Baseline | - |
| DoG | Structural | Gradient suppression |
| Retinex (SSR) | Frequency | Illumination/reflectance |
| MSR | Multi-frequency | Broad illumination |
| Homomorphic | Frequency | Illumination suppression |
| Shadow removal | Morphological | Cast shadow |

**Protocol**:
1. Fix detector to SIFT (baseline)
2. Apply each illumination method
3. Compare RMSE and inlier_ratio improvement vs baseline

### E3: Scale Invariance

| Pair Type | Scale Ratio | Expected Difficulty |
|-----------|------------|---------------------|
| OHRC ↔ OHRC | 1:1 | Easy (same sensor) |
| OHRC ↔ NAC | 1:2 | Moderate |
| OHRC ↔ TMC-2 | 1:20 | Hard |
| NAC ↔ TMC-2 | 1:10 | Hard |

**Protocol**:
1. Use SIFT + Retinex (best from E2)
2. Evaluate matching across resolution ratios
3. Measure degradation of RMSE with increasing scale difference

### E4: Geometry Estimation

| Method | Type | Expected Benefit |
|--------|------|-----------------|
| RANSAC | Standard | Baseline |
| MAGSAC++ | Adaptive threshold | Better outlier handling |
| LO-RANSAC | Local optimization | Higher inlier count |

**Protocol**:
1. Fix detector + preprocessing (best from E1/E2)
2. Compare geometry methods
3. Measure inlier ratio improvement

### E5: Refinement

| Method | Type | Target |
|--------|------|--------|
| None | Baseline | - |
| Template matching | Image-based | Sub-pixel alignment |
| Phase correlation | Frequency | Translation correction |
| IRLS | Statistical | Robust refit |
| Combined (all) | Sequential | Best possible |

**Protocol**:
1. Use best pipeline from E1-E4
2. Apply each refinement method
3. Measure RMSE improvement

## Metrics

| Metric | Definition | Better |
|--------|-----------|--------|
| RMSE | √(mean squared reprojection error) | Lower |
| Inlier ratio | #inliers / #matches | Higher |
| Spatial coverage | Fraction of grid cells with ≥1 match | Higher |
| Processing time | Total pipeline time (seconds) | Lower |

## Running Experiments

```bash
# Single experiment
python -m lunar_registration.cli register \
    --source data/external/ch2/pilot/ch2_ohrc_001.tif \
    --reference data/external/lro/pilot/lro_nac_001.tif \
    --detector sift --illumination retinex

# Full benchmark
python -m lunar_registration.cli benchmark \
    --pairs data/manifests/benchmark_pairs.json \
    --detectors sift orb \
    --output outputs/benchmark
```

## Expected Results

Based on literature for similar lunar/planetary registration tasks:

| Configuration | Expected RMSE | Expected Coverage |
|--------------|---------------|-------------------|
| SIFT baseline | 3–8 px | 40–60% |
| SIFT + Retinex | 2–5 px | 50–70% |
| SuperPoint + MSR | 1–3 px | 60–80% |
| LoFTR + full refinement | 0.5–2 px | 70–90% |
