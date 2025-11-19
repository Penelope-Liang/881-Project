# Step 1 - Data scan and pairing validation

- Scripts: scan_and_validate.py, test_scan_outputs.py
- What it does:
  - Count patients/XML/DICOM and sample XMLs
  - Parse XML ROI and match to DICOM by SOP UID
  - Render overlays (red polygon) on CT slices
  - Write summary.json and missing_pairs.json
- How to run:

`bash
python step1/scan_and_validate.py --data-root LIDC/dataset/LIDC-IDRI --samples 5 --out outputs/scan
python step1/test_scan_outputs.py --out outputs/scan
`