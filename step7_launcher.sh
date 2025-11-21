#!/bin/bash

# Step7 Intelligent CT Analysis Pipeline Launcher

echo "============================================"
echo "Step7: Intelligent CT Analysis Pipeline"
echo "============================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check setup
echo "Checking pipeline setup..."
python -m step7.test_setup --check

echo ""
echo "Available options:"
echo "1. Run complete intelligent pipeline"
echo "2. Test pipeline components"
echo "3. Launch Slicer with existing data"
echo ""

read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo "Running intelligent pipeline..."
        echo "Using test data: outputs/step6_case_3d/ct_volume.nii.gz"
        
        python -m step7.intelligent_pipeline \
            --input outputs/step6_case_3d/ct_volume.nii.gz \
            --unet outputs/step4/unet_best.path \
            --regression outputs/step5_reg/reg_best.path \
            --output outputs/step7_analysis
        ;;
    2)
        echo "Testing pipeline components..."
        python -m step7.test_setup --command
        ;;
    3)
        echo "Launching Slicer with existing data..."
        ./run_slicer_test.sh
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Step7 Pipeline Complete!"
echo "============================================"