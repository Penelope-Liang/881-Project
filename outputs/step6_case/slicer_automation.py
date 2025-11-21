import json
import slicer
import slicer.util

ct_volume_path = r"outputs/step6_case/ct_volume.nii.gz"
seg_path = r"outputs/step6_case/segmentation_volume.nii.gz"
slice_info = json.loads(r"""[{"slice_index": 0, "predicted_diameter_mm": 48.37664794921875, "predicted_category": ">10mm"}]""")

volume_node = slicer.util.loadVolume(ct_volume_path)
label_node = slicer.util.loadLabelVolume(seg_path)
label_display = label_node.GetDisplayNode()
if label_display:
    label_display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRed")
    label_display.SetOpacity(0.5)
slicer.util.setSliceViewerLayers(background=volume_node, label=label_node, labelOpacity=0.5)

segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "NoduleSegmentation")
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label_node, segmentation_node)
label_node.SetDisplayVisibility(False)

dims = volume_node.GetImageData().GetDimensions()
center_i = dims[0] // 2
center_j = dims[1] // 2

def jump_to_slice(index):
    layout_manager = slicer.app.layoutManager()
    red_slice_widget = layout_manager.sliceWidget("Red")
    if red_slice_widget:
        slice_logic = red_slice_widget.sliceLogic()
        spacing = volume_node.GetSpacing()
        origin = volume_node.GetOrigin()
        slice_offset = origin[2] + index * spacing[2]
        slice_logic.SetSliceOffset(slice_offset)
        print(f"[Slicer] Set Red view slice offset to {slice_offset:.2f} mm (index {index})")

if slice_info:
    target_idx = int(slice_info[0]["slice_index"])
    jump_to_slice(target_idx)
    
    print(f"[Slicer] Jumped to slice index: {target_idx}")
    print(f"[Slicer] Predicted diameter: {slice_info[0]['predicted_diameter_mm']:.2f} mm")
    print(f"[Slicer] Predicted category: {slice_info[0]['predicted_category']}")
    
    fiducials = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PredictedSlices")
    for item in slice_info:
        idx = int(item["slice_index"])
        ras = [0.0, 0.0, 0.0]
        volume_node.TransformIndexToPhysicalPoint([center_i, center_j, idx], ras)
        label = f'Slice {idx}: {item["predicted_category"]} ({item["predicted_diameter_mm"]:.1f} mm)'
        fiducials.AddControlPoint(ras, label)
    
    text_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", "SliceInfo")
    text_node.SetText(f"Target Slice Index: {target_idx}\nPredicted: {slice_info[0]['predicted_category']} ({slice_info[0]['predicted_diameter_mm']:.1f} mm)")
    print("[Slicer] Text annotation added: 'SliceInfo' node")
else:
    print("No slice info available to jump to")

crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
if crosshair:
    crosshair.SetCrosshairMode(crosshair.ShowBasic)

layout_manager = slicer.app.layoutManager()
red_slice_widget = layout_manager.sliceWidget("Red")
if red_slice_widget:
    slice_logic = red_slice_widget.sliceLogic()
    offset = slice_logic.GetSliceOffset()
    spacing = volume_node.GetSpacing()
    origin = volume_node.GetOrigin()
    calculated_index = int(round((offset - origin[2]) / spacing[2]))
    print(f"[Slicer] Current slice offset: {offset:.2f} mm")
    print(f"[Slicer] Calculated slice index: {calculated_index}")
    if slice_info:
        print(f"[Slicer] Target slice index: {target_idx}")
        if calculated_index == target_idx:
            print("[Slicer] ✓ Correct slice index confirmed!")
        else:
            print(f"[Slicer] ⚠ Warning: Slice index mismatch (expected {target_idx}, got {calculated_index})")

print("[Slicer] To verify slice index:")
print("[Slicer]   1. Check the 'PredictedSlices' fiducial marker label")
print("[Slicer]   2. Check the 'SliceInfo' text node in Data module")
print("[Slicer]   3. Look at the slice offset value in the Red view control bar")

print("[Slicer] Setting up 3D visualization...")
try:
    volume_rendering_node = slicer.modules.volumerendering.logic().CreateVolumeRenderingNode(volume_node)
    if volume_rendering_node:
        volume_rendering_display_node = volume_rendering_node.GetDisplayNode()
        if volume_rendering_display_node:
            volume_rendering_display_node.SetVisibility(True)
            print("[Slicer] 3D volume rendering enabled")
    
    segmentation_display_node = segmentation_node.GetDisplayNode()
    if segmentation_display_node:
        segmentation_display_node.SetVisibility3D(True)
        segmentation_display_node.SetOpacity3D(0.7)
        print("[Slicer] 3D segmentation visualization enabled")
    
    layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    print("[Slicer] Layout set to Four-Up view (2D slices + 3D view)")
    print("[Slicer] You can now see:")
    print("[Slicer]   - 2D slices in Red/Yellow/Green views")
    print("[Slicer]   - 3D volume rendering in the 3D view (top-right)")
    print("[Slicer]   - 3D segmentation overlay in the 3D view")
except Exception as e:
    print(f"[Slicer] Note: 3D rendering setup had issues (this is optional): {e}")
    print("[Slicer] You can manually enable 3D view in the Layout menu")
