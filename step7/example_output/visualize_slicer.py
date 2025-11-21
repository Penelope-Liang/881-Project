"""
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
summary_path = r"/Users/penelopel/Desktop/881_project/step7/example_output/summary.json"
with open(summary_path, "r") as f:
    summary = json.load(f)

top1_idx = summary["top1_slice_index"]
nodules = summary["nodules"]

print(f"Top-1 Slice: {top1_idx}")
print(f"Nodules: {len(nodules)}")

# 1. load ct volume
print("\n1. Loading CT volume...")
volume_path = r"/Users/penelopel/Desktop/881_project/step7/example_output/series.nii.gz"
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
label_path = r"/Users/penelopel/Desktop/881_project/step7/example_output/series_label.nii.gz"
ln = slicer.util.loadLabelVolume(label_path)
ln.SetName("Nodule_Segmentation")

label_display = ln.GetDisplayNode()
if label_display:
    label_display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRed")
    label_display.SetOpacity(0.5)

# create 3d segmentation model
print("3. Creating 3D segmentation model...")
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
print("4. Creating annotations...")
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
    centroidNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", 
                                                       f"Nodule_{nod_idx+1}_Centroid")
    fidIdx = centroidNode.AddControlPoint(ras_center[0], ras_center[1], ras_center[2])
    centroidNode.SetNthControlPointLabel(fidIdx, f"{nod['diameter_mm']:.1f}mm")
    
    centroidDisplay = centroidNode.GetDisplayNode()
    if centroidDisplay:
        centroidDisplay.SetSelectedColor(1.0, 1.0, 0.0)
        centroidDisplay.SetGlyphScale(3.5)
        centroidDisplay.SetTextScale(4.5)
        centroidDisplay.SetVisibility(1)
        centroidDisplay.SetVisibility3D(1)
    
    # bounding box (green roi)
    roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", 
                                                  f"Nodule_{nod_idx+1}_BBox")
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
    curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode",
                                                    f"Nodule_{nod_idx+1}_Contour")
    
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
print("5. Setting layout...")
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
print("6. Configuring 3D view...")
threeDWidget = layoutManager.threeDWidget(0)
if threeDWidget:
    threeDView = threeDWidget.threeDView()
    threeDView.resetFocalPoint()
    threeDView.resetCamera()

print("\n" + "="*80)
print("Visualization Complete!")
print("="*80)
print("Annotations:")
print("  - YELLOW: Centroids with diameter labels")
print("  - GREEN: Bounding boxes")
print("  - CYAN: Contours")
print("  - RED: 3D segmentation model")
print("="*80)
