#!/bin/bash

# Script to run 3D Slicer with test visualization

echo "============================================"
echo "3D Slicer Test Script"
echo "============================================"

# Check if output directory exists
OUTPUT_DIR="outputs/step6_case_3d"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    echo "Please run the pipeline first:"
    echo "  python -m step6.integrated_pipeline --ct-dir ... --out outputs/step6_case_3d"
    exit 1
fi

# Check if files exist
CT_FILE="$OUTPUT_DIR/ct_volume.nii.gz"
SEG_FILE="$OUTPUT_DIR/segmentation_volume.nii.gz"

if [ ! -f "$CT_FILE" ]; then
    echo "Error: CT volume not found: $CT_FILE"
    exit 1
fi

if [ ! -f "$SEG_FILE" ]; then
    echo "Error: Segmentation volume not found: $SEG_FILE"
    exit 1
fi

echo "Found required files:"
echo "  - $CT_FILE"
echo "  - $SEG_FILE"
echo ""

# Use provided path or find Slicer installation
SLICER_PATH="$1"

if [ -z "$SLICER_PATH" ]; then
    echo "Searching for 3D Slicer installation..."
    
    # Search in common locations
    SEARCH_PATHS=(
        "/Applications/Slicer.app/Contents/MacOS/Slicer"
        "/Applications/Slicer-5.app/Contents/MacOS/Slicer"
        "/Applications/Slicer-5.6.app/Contents/MacOS/Slicer"
        "/Applications/Slicer 5.6.app/Contents/MacOS/Slicer"
        "$HOME/Applications/Slicer.app/Contents/MacOS/Slicer"
        "/Applications/3D Slicer.app/Contents/MacOS/Slicer"
    )
    
    for path in "${SEARCH_PATHS[@]}"; do
        if [ -f "$path" ]; then
            SLICER_PATH="$path"
            break
        fi
    done
    
    # Try to use 'open' command as fallback
    if [ -z "$SLICER_PATH" ]; then
        echo ""
        echo "Auto-search failed. Trying to find Slicer.app..."
        SLICER_APP=$(find /Applications -name "Slicer*.app" -o -name "*Slicer*.app" 2>/dev/null | head -1)
        
        if [ -n "$SLICER_APP" ]; then
            SLICER_PATH="$SLICER_APP/Contents/MacOS/Slicer"
            echo "Found: $SLICER_APP"
        fi
    fi
fi

if [ -z "$SLICER_PATH" ] || [ ! -f "$SLICER_PATH" ]; then
    echo ""
    echo "============================================"
    echo "ERROR: 3D Slicer executable not found!"
    echo "============================================"
    echo ""
    echo "Please specify the path manually:"
    echo "  ./run_slicer_test.sh /path/to/Slicer.app/Contents/MacOS/Slicer"
    echo ""
    echo "To find your Slicer location:"
    echo "  1. Open Finder"
    echo "  2. Find Slicer application"
    echo "  3. Right-click -> Show Package Contents"
    echo "  4. Navigate to: Contents/MacOS/Slicer"
    echo ""
    echo "Or just open Slicer manually and run in Python console:"
    echo "  exec(open('step6/test_3d_simple.py').read())"
    echo ""
    exit 1
fi

echo "Found Slicer at: $SLICER_PATH"
echo ""
echo "Launching 3D Slicer..."
echo ""

# Run Slicer with the intelligent pipeline script
"$SLICER_PATH" --python-script step7/slicer_utils.py

echo ""
echo "============================================"
echo "Done!"
echo "============================================"
