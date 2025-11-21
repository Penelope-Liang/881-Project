import os
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from plan_b.slicer.visualize import generate_slicer_script, launch_slicer


def test_visualization():
    """
    test automated visualization with example output
    """
    print("Testing Automated 3D Slicer Visualization")
    
    
    # paths
    script_dir = Path(__file__).parent
    result_dir = script_dir / "example_output"
    slicer_script_path = result_dir / "test_visualize.py"
    slicer_executable = "D:/Slicer 5.8.1/Slicer.exe"
    
    # check if example output exists
    print("1. Checking example output files")
    required_files = ["series.nii.gz", "series_label.nii.gz", "summary.json"]
    
    for file in required_files:
        file_path = result_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"{file}: {size_mb:.1f} MB")
        else:
            print(f"{file}: NOT FOUND")
            return False
    
    # generate slicer script
    print("2. Generating Slicer visualization script")
    try:
        generate_slicer_script(result_dir, slicer_script_path)
        print(f"Script generated: {slicer_script_path}")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # check if slicer executable exists
    print("\n3. Checking 3D Slicer installation...")
    if os.path.exists(slicer_executable):
        print(f"Slicer found: {slicer_executable}")
    else:
        print(f"Slicer not found: {slicer_executable}")
        print("Please update the path in this script or use --slicer-path argument")
        return False
    
    # launch slicer
    print("\n4. Launching 3D Slicer...")
    try:
        launch_slicer(slicer_executable, slicer_script_path)
        print("Slicer launched successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print("Test Complete!")
    print("\nExpected visualization:")
    print("Four-view layout (Axial, Sagittal, Coronal, 3D)")
    print("CT volume with lung window")
    print("Red segmentation overlay")
    print("Yellow centroids with diameter labels")
    print("Green bounding boxes")
    print("Cyan contours")
    print("Red 3D nodule model")
    return True


if __name__ == "__main__":
    success = test_visualization()
    sys.exit(0 if success else 1)
