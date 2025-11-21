#!/bin/bash
# test Automated 3D Slicer Visualization
# this script demonstrates the automated visualization workflow


echo "Testing Automated 3D Slicer Visualization"


# change to project root
cd "$(dirname "$0")/../.."

echo ""
echo "Step 1: Verify example output files..."
ls -lh plan_b/3d_slicer/example_output/

echo ""
echo "Step 2: Generate Slicer visualization script..."
python plan_b/3d_slicer/visualize.py \
    --result-dir plan_b/3d_slicer/example_output \
    --slicer-path "D:/Slicer 5.8.1/Slicer.exe" \
    --output-script plan_b/3d_slicer/example_output/auto_visualize.py

echo ""
echo "Test Complete!"
echo ""
echo "3D Slicer should now be opening with the visualization."
echo "Expected display:"
echo "Four-view layout (Axial, Sagittal, Coronal, 3D)"
echo "CT volume with optimized lung window"
echo "Red segmentation overlay"
echo "Yellow centroids with diameter labels"
echo "Green bounding boxes"
echo "Cyan contours"
echo "Red 3D nodule surface model"