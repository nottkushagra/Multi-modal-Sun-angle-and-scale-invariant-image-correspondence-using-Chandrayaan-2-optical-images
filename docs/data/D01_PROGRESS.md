# D-01: Pilot Data Acquisition — Progress Tracker

## Task Summary

Acquire a small pilot dataset (3–5 image pairs) to validate the full
SelenoFuse pipeline before scaling to the complete dataset.

## Checklist

### 1. Portal Access
- [ ] Register at [ISRO PRADAN](https://pradan.issdc.gov.in)
- [ ] Verify access to Chandrayaan-2 OHRC / TMC-2 L2 products
- [ ] Register at [LROC QuickMap](https://quickmap.lroc.asu.edu)

### 2. Image Selection
- [ ] Identify target region in Mare Imbrium (~10 km × 10 km)
- [ ] Select 2–3 OHRC images at different sun elevations (15°, 45°, 70°)
- [ ] Select corresponding TMC-2 images for cross-scale pairing
- [ ] Select 2–3 LRO NAC images at the same region for cross-sensor pairing

### 3. Download & Organize
- [ ] Download CH2 images → `data/external/ch2/pilot/`
- [ ] Download LRO images → `data/external/lro/pilot/`
- [ ] Populate `data/manifests/pilot_manifest.csv` with real metadata

### 4. Validation
- [ ] Verify all images load correctly with `load_image()`
- [ ] Run baseline pipeline on at least 1 pair
- [ ] Update `docs/data/PILOT_DATA_REPORT.md` with findings

## Status

| Step | Status | Date | Notes |
|------|--------|------|-------|
| Portal access | ⬜ Pending | | |
| Image selection | ⬜ Pending | | |
| Download | ⬜ Pending | | |
| Validation | ⬜ Pending | | |
