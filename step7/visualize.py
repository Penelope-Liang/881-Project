import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path


def find_slicer_executable():
    """
    automatically find 3d slicer executable path
    
    returns:
        path to slicer executable if found, otherwise None
    """
    system = platform.system()
    
    # common slicer paths by platform
    common_paths = []
    
    if system == "Darwin":  # mac
        common_paths = [
            "/Applications/Slicer.app/Contents/MacOS/Slicer",
            "/Applications/Slicer-5.8.1.app/Contents/MacOS/Slicer",
            "/Applications/Slicer-5.7.app/Contents/MacOS/Slicer",
        ]
    elif system == "Windows":  # windows
        common_paths = [
            "D:/Slicer 5.8.1/Slicer.exe",
            "C:/Program Files/Slicer 5.8.1/Slicer.exe",
            "C:/Program Files/Slicer/Slicer.exe",
        ]
    elif system == "Linux":  # linux
        common_paths = [
            "/usr/local/Slicer/Slicer",
            "/opt/Slicer/Slicer",
            os.path.expanduser("~/Slicer/Slicer"),
        ]
    
    # check common paths
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # try to find via command line (which/where)
    try:
        if system == "Windows":
            result = subprocess.run(["where", "Slicer.exe"], 
                                  capture_output=True, text=True, timeout=2)
        else:
            result = subprocess.run(["which", "Slicer"], 
                                  capture_output=True, text=True, timeout=2)
        
        if result.returncode == 0:
            found_path = result.stdout.strip().split("\n")[0]
            if os.path.exists(found_path):
                return found_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return None


def generate_slicer_script(result_dir, slicer_script_path):
    """
    generate 3d slicer python script for visualization
    
    args:
        result_dir: directory containing pipeline results
        slicer_script_path: output path for generated slicer script
    """
    result_dir = Path(result_dir).absolute()
    
    script = f'''"""
automated 3D Slicer with script
"""
import slicer
import json
import vtk
import numpy as np

print("="*80)
print("3D Slicer Visualization")
print("="*80)

# clear scene
slicer.mrmlScene.Clear(0)

# load summary
summary_path = r"{result_dir / "summary.json"}"
with open(summary_path, "r") as f:
    summary = json.load(f)

top1_idx = summary["top1_slice_index"]
nodules = summary["nodules"]

print(f"Top-1 Slice: {{top1_idx}}")
print(f"Nodules: {{len(nodules)}}")

# 1. load ct volume
print("\\n1. Loading CT volume...")
volume_path = r"{result_dir / "series.nii.gz"}"
vn = slicer.util.loadVolume(volume_path)
vn.SetName("CT_Volume")

# set optimized window/level for lung viewing
display = vn.GetDisplayNode()
if display:
    display.SetAutoWindowLevel(0)
    display.SetWindow(2000)
    display.SetLevel(-400)
    print("  Window/Level: W=2000, L=-400")

# 2. load segmentation
print("2. Loading segmentation")
label_path = r"{result_dir / "series_label.nii.gz"}"
ln = slicer.util.loadLabelVolume(label_path)
ln.SetName("Nodule_Segmentation")

label_display = ln.GetDisplayNode()
if label_display:
    label_display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRed")
    label_display.SetOpacity(0.5)

# create 3d segmentation model
print("3. Creating 3D segmentation model")
segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Nodule_3D")
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(ln, segNode)
segNode.CreateClosedSurfaceRepresentation()
segNode.SetReferenceImageGeometryParameterFromVolumeNode(vn)

segDisplay = segNode.GetDisplayNode()
if segDisplay:
    segDisplay.SetOpacity3D(0.8)
    segDisplay.SetVisibility(1)
    segDisplay.SetVisibility3D(1)

# get transform
ijkToRas = vtk.vtkMatrix4x4()
vn.GetIJKToRASMatrix(ijkToRas)
spacing = vn.GetSpacing()

# 4. create annotations for each nodule
print("4. Creating annotations")
for nod_idx, nod in enumerate(nodules):
    bbox = nod["bbox"]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    ijk_center = [cx, cy, top1_idx, 1]
    ras_center = ijkToRas.MultiplyPoint(ijk_center)
    
    # centroid (yellow point)
    centroidNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"Nodule_{{nod_idx+1}}_Centroid")
    fidIdx = centroidNode.AddControlPoint(ras_center[0], ras_center[1], ras_center[2])
    centroidNode.SetNthControlPointLabel(fidIdx, f"{{nod['diameter_mm']:.1f}}mm")
    
    centroidDisplay = centroidNode.GetDisplayNode()
    if centroidDisplay:
        centroidDisplay.SetSelectedColor(1.0, 1.0, 0.0)
        centroidDisplay.SetGlyphScale(3.5)
        centroidDisplay.SetTextScale(4.5)
        centroidDisplay.SetVisibility(1)
        centroidDisplay.SetVisibility3D(1)
    
    # bounding box (green roi)
    roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", f"Nodule_{{nod_idx+1}}_BBox")
    roiNode.SetCenter(ras_center[0], ras_center[1], ras_center[2])
    roiNode.SetSize(width * spacing[0], height * spacing[1], spacing[2] * 3)
    
    roiDisplay = roiNode.GetDisplayNode()
    if roiDisplay:
        roiDisplay.SetSelectedColor(0.0, 1.0, 0.0)
        roiDisplay.SetFillOpacity(0.0)
        roiDisplay.HandlesInteractiveOff()
        roiDisplay.SetVisibility(1)
        roiDisplay.SetVisibility3D(1)
    
    # contour (cyan curve)
    curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode", f"Nodule_{{nod_idx+1}}_Contour")
    
    num_points = 20
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        offset_x = (width / 2.0) * np.cos(angle)
        offset_y = (height / 2.0) * np.sin(angle)
        
        ijk_point = [cx + offset_x, cy + offset_y, top1_idx, 1]
        ras_point = ijkToRas.MultiplyPoint(ijk_point)
        curveNode.AddControlPoint(ras_point[0], ras_point[1], ras_point[2])
    
    curveDisplay = curveNode.GetDisplayNode()
    if curveDisplay:
        curveDisplay.SetSelectedColor(0.0, 1.0, 1.0)
        curveDisplay.SetLineWidth(2.5)
        curveDisplay.SetVisibility(1)
        curveDisplay.SetVisibility3D(1)

# 5. set layout to four-view
print("5. Setting layout")
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)

# 6. configure all 2d views
layoutManager = slicer.app.layoutManager()
for viewName in ["Red", "Yellow", "Green"]:
    widget = layoutManager.sliceWidget(viewName)
    if widget:
        sliceLogic = widget.sliceLogic()
        compositeNode = sliceLogic.GetSliceCompositeNode()
        compositeNode.SetBackgroundVolumeID(vn.GetID())
        compositeNode.SetLabelVolumeID(ln.GetID())
        compositeNode.SetLabelOpacity(0.5)
        sliceLogic.FitSliceToAll()

# 7. configure 3d view
print("6. Configuring 3D view")
threeDWidget = layoutManager.threeDWidget(0)
if threeDWidget:
    threeDView = threeDWidget.threeDView()
    threeDView.resetFocalPoint()
    threeDView.resetCamera()

print("\\n" + "="*80)
print("Visualization Complete!")
print("Annotations:")
print("YELLOW: Centroids with diameter labels")
print("GREEN: Bounding boxes")
print("CYAN: Contours")
print("RED: 3D segmentation model")
'''
    
    # write script
    with open(slicer_script_path, "w", encoding="utf-8") as f:
        f.write(script)
    
    print(f"Slicer script generated: {slicer_script_path}")


def launch_slicer(slicer_executable, slicer_script_path):
    """
    launch 3d slicer with the generated script
    
    args:
        slicer_executable: path to slicer executable
        slicer_script_path: path to slicer python script
    """
    print(f"\nLaunching 3D Slicer...")
    print(f"  Executable: {slicer_executable}")
    print(f"  Script: {slicer_script_path}")
    
    cmd = [slicer_executable, "--python-script", str(slicer_script_path)]
    
    try:
        subprocess.Popen(cmd)
        print("\n3D Slicer launched successfully!")
    except Exception as e:
        print(f"\nError launching Slicer: {e}")
        print("\nManual launch command:")
        print(f"  \"{slicer_executable}\" --python-script \"{slicer_script_path}\"")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and launch 3D Slicer visualization")
    parser.add_argument("--result-dir", type=str, required=True, help="Directory containing pipeline results")
    parser.add_argument("--slicer-path", type=str, default=None, help="Path to 3D Slicer executable (auto-detected if not specified)")
    parser.add_argument("--output-script", type=str, default=None, help="Output path for generated Slicer script (default: result_dir/visualize.py)")
    
    args = parser.parse_args()
    
    # auto-detect slicer path if not specified
    if args.slicer_path is None:
        args.slicer_path = find_slicer_executable()
        if args.slicer_path:
            print(f"Auto-detected Slicer path: {args.slicer_path}")
        else:
            print("\nError: Could not auto-detect 3D Slicer installation.")
            print("Please specify the path with --slicer-path")
            print("\nCommon paths:")
            system = platform.system()
            if system == "Darwin":
                print("  Mac: /Applications/Slicer.app/Contents/MacOS/Slicer")
            elif system == "Windows":
                print("  Windows: D:/Slicer 5.8.1/Slicer.exe")
            elif system == "Linux":
                print("  Linux: /usr/local/Slicer/Slicer")
            sys.exit(1)
    
    # default output script path
    if args.output_script is None:
        args.output_script = Path(args.result_dir) / "visualize_slicer.py"
    
    # generate script
    generate_slicer_script(args.result_dir, args.output_script)
    
    # launch slicer
    if os.path.exists(args.slicer_path):
        launch_slicer(args.slicer_path, args.output_script)
    else:
        print(f"\nWarning: Slicer executable not found at: {args.slicer_path}")
        print("Please specify the correct path with --slicer-path")
        print(f"\nLaunch manually:")
        print(f"\"{args.slicer_path}\" --python-script \"{args.output_script}\"")
