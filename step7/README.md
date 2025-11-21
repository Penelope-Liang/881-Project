# Step 7 - 3D Slicer Visualization Pipeline
- Scripts: pipeline.py, visualize.py
- What it does:
  - Run end-to-end nodule detection and segmentation pipeline
  - Generate 3D NIfTI volumes and segmentation masks
  - Automatically launch 3D Slicer with annotations

- How to run:
```bash
# step 1: run detection and segmentation pipeline
# note: replace <placeholders> with actual paths before running
python step7/pipeline.py \
    --ct-dir <path_to_dicom_directory> \
    --query ">3mm" \
    --detector <path_to_yolo_model> \
    --unet <path_to_unet_model> \
    --regressor <path_to_regression_model> \
    --output-dir <output_directory> \
    --device cpu

# step 2: generate Slicer script and launch visualization
# slicer path is auto-detected, or specify manually with --slicer-path
python step7/visualize.py \
    --result-dir <output_directory>
```

See main README.md for detailed instructions and examples.