# 3D Slicer automation script
# Automatically import CT volume and segmentation mask, navigate to most relevant slice,
# highlight tumor regions

import slicer
import json
import os

def main():
	print("=" * 60)
	print("Step7: Intelligent CT Analysis Pipeline - 3D Slicer Automation")
	print("=" * 60)
	
	# Load metadata
	metadata_path = r"outputs/step7_test/metadata.json"
	with open(metadata_path, 'r') as f:
		metadata = json.load(f)
	
	best_slice_idx = metadata['best_slice_index']
	total_slices = metadata['total_slices']
	best_diameter = metadata['best_diameter_mm']
	
	print(f"Most relevant slice index: {best_slice_idx}/{total_slices}")
	print(f"Predicted diameter: {best_diameter:.2f}mm")
	
	# Clear scene
	slicer.mrmlScene.Clear()
	
	# Set four-up layout
	layout_manager = slicer.app.layoutManager()
	layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
	
	# Load CT volume
	ct_volume_path = r"outputs/step7_test/ct_volume.nii.gz"
	print(f"Loading CT volume: {ct_volume_path}")
	ct_node = slicer.util.loadVolume(ct_volume_path)
	ct_node.SetName("CT_Volume")
	
	# Set CT display window/level (suitable for lung CT)
	ct_display = ct_node.GetDisplayNode()
	if ct_display:
		ct_display.SetWindow(1500)
		ct_display.SetLevel(-500)
		print("Set CT display window/level: Window=1500, Level=-500")
	
	# Load segmentation mask as label volume
	seg_volume_path = r"outputs/step7_test/segmentation_volume.nii.gz"
	print(f"Loading segmentation mask: {seg_volume_path}")
	
	# Use loadLabelVolume to load mask as label volume for overlay display
	label_node = slicer.util.loadLabelVolume(seg_volume_path)
	label_node.SetName("Segmentation_Mask")
	
	# Set label display properties
	label_display = label_node.GetDisplayNode()
	if label_display:
		# Set color table to red
		label_display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRed")
		# Set opacity for overlay
		label_display.SetOpacity(0.5)
		print("Set segmentation mask display: red color, opacity 0.5")
	
	# Set slice viewer layers to show CT as background and mask as overlay
	slicer.util.setSliceViewerLayers(background=ct_node, label=label_node, labelOpacity=0.5)
	print("Set slice viewer layers: CT as background, mask as overlay")
	
	# Navigate to most relevant slice
	print(f"Navigating to most relevant slice: {best_slice_idx}")
	
	# Get volume spacing and bounds to calculate slice position
	volume_spacing = ct_node.GetSpacing()
	slice_spacing = abs(volume_spacing[2]) if volume_spacing[2] != 0 else 2.0
	
	bounds = [0, 0, 0, 0, 0, 0]
	ct_node.GetBounds(bounds)
	z_min = bounds[4]
	
	# Calculate target Z position using origin and slice spacing
	volume_origin = ct_node.GetOrigin()
	target_z = volume_origin[2] + (best_slice_idx * slice_spacing)
	
	print(f"Target Z position: {target_z:.2f}mm (slice spacing: {slice_spacing:.2f}mm)")
	
	# Navigate to target slice in all views
	layout_manager = slicer.app.layoutManager()
	for view_name in ["Red", "Yellow", "Green"]:
		slice_widget = layout_manager.sliceWidget(view_name)
		if slice_widget:
			slice_logic = slice_widget.sliceLogic()
			slice_logic.SetSliceOffset(target_z)
			if view_name == "Red":
				print(f"Set Axial view (Red) to Z position: {target_z:.2f}mm")
	
	# Ensure red view is active and showing the slice
	red_slice_widget = layout_manager.sliceWidget("Red")
	if red_slice_widget:
		red_slice_widget.sliceLogic().SetSliceOffset(target_z)
		# Fit slice to window
		red_slice_widget.fitSliceToBackground()
	
	# Reset slice views to ensure proper display
	slicer.util.resetSliceViews()
	
	# Print completion message
	print("=" * 60)
	print("Automation complete!")
	print("=" * 60)
	print(f"CT Volume loaded: {ct_node.GetName()}")
	print(f"Segmentation mask loaded: {label_node.GetName()}")
	print(f"Navigated to most relevant slice: {best_slice_idx}/{total_slices}")
	print(f"Predicted diameter: {best_diameter:.2f}mm")
	print(f"Tumor regions highlighted (red overlay)")
	print("=" * 60)

if __name__ == "__main__":
	main()
