# SelenoFuse — Dataset Documentation

## Data Sources

### Chandrayaan-2

| Sensor | Resolution | Swath | Product Level |
|--------|-----------|-------|---------------|
| **OHRC** (Orbiter High Resolution Camera) | 0.25 m/px | 3 km | L2 (radiometrically calibrated) |
| **TMC-2** (Terrain Mapping Camera) | 5.0 m/px | 20 km | L2 |

**Access portal**: [ISRO PRADAN](https://pradan.issdc.gov.in)

#### Download Instructions

1. Register at https://pradan.issdc.gov.in
2. Navigate to **Chandrayaan-2 → OHRC** or **TMC-2**
3. Use the map interface to select your region of interest
4. Filter by:
   - Product type: L2
   - Sun elevation: record this for illumination experiments
   - Cloud cover: N/A for lunar
5. Download as GeoTIFF

### LRO (Lunar Reconnaissance Orbiter)

| Sensor | Resolution | Swath | Product Level |
|--------|-----------|-------|---------------|
| **NAC** (Narrow Angle Camera) | 0.5 m/px | 5 km | CDR (calibrated data record) |

**Access portal**: [LROC QuickMap](https://quickmap.lroc.asu.edu)

#### Download Instructions

1. Go to https://quickmap.lroc.asu.edu
2. Search for your target region (same lat/lon as CH2 data)
3. Select NAC images
4. Download from PDS: https://pds.lroc.asu.edu

## Data Organization

```
data/
├── external/
│   ├── ch2/
│   │   └── pilot/          # Chandrayaan-2 pilot dataset
│   │       ├── ch2_ohrc_001.tif
│   │       └── ch2_tmc2_001.tif
│   └── lro/
│       └── pilot/          # LRO pilot dataset
│           └── lro_nac_001.tif
├── manifests/
│   └── pilot_manifest.csv  # Image metadata
├── reference/              # Reference images
└── source/                 # Source images for testing
```

## Manifest Format

The manifest CSV tracks all images with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | str | Unique identifier |
| `sensor` | str | OHRC, TMC2, or NAC |
| `product_type` | str | L2 or CDR |
| `resolution_m` | float | Ground sampling distance (m/px) |
| `sun_elevation_deg` | float | Sun elevation angle at acquisition |
| `phase_angle_deg` | float | Phase angle |
| `center_lat` | float | Scene center latitude |
| `center_lon` | float | Scene center longitude |
| `file_path` | str | Relative path to image file |
| `notes` | str | Free-form notes |

## Experiment Configurations

### Illumination Robustness

Pairs with the same region but different sun elevations:
- Low (10–25°) vs High (60–80°)
- Extreme low (< 10°) vs Moderate (30–50°)

### Scale Invariance

Cross-resolution pairs:
- OHRC (0.25 m) ↔ TMC-2 (5 m): 20× scale difference
- OHRC (0.25 m) ↔ NAC (0.5 m): 2× scale difference

### Cross-Sensor

Different instruments imaging the same region:
- CH2 OHRC ↔ LRO NAC
- CH2 TMC-2 ↔ LRO NAC
