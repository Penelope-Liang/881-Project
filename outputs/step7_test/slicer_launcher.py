#!/usr/bin/env python3
"""
3D Slicer Automation Script - Integrated Pipeline
Automatically loads CT, segmentation, and jumps to best slice
"""

import slicer
import numpy as np
import vtk
import os

# Configuration - using absolute paths
CT_VOLUME_PATH = r"/Users/penelopel/Desktop/881_project/outputs/step7_test/ct_volume.nii.gz"
SEGMENTATION_PATH = r"/Users/penelopel/Desktop/881_project/outputs/step7_test/segmentation_volume.nii.gz"
BEST_SLICE_INDEX = 173
WINDOW = 1500.0
LEVEL = -550.0
LABEL_OPACITY = 0.9

def main():
    print("=" * 60)
    print("Integrated Pipeline - 3D Slicer Automation")
    print("=" * 60)
    
    # Clear scene
    slicer.mrmlScene.Clear()
    
    # Set layout to Four-Up view
    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    
    # Load CT volume
    print("Loading CT volume...")
    ct_node = slicer.util.loadVolume(CT_VOLUME_PATH)
    ct_node.SetName("CT_Volume")
    
    # Apply window/level settings for lung visualization
    ct_display = ct_node.GetDisplayNode()
    if ct_display:
        ct_display.SetWindow(WINDOW)
        ct_display.SetLevel(LEVEL)
        print("Applied window/level: Window={}, Level={}".format(WINDOW, LEVEL))
    
    # Load segmentation as label volume
    print("Loading segmentation mask...")
    label_node = slicer.util.loadLabelVolume(SEGMENTATION_PATH)
    label_node.SetName("Tumor_Segmentation")
    
    # Configure label display (red highlight for tumor)
    label_display = label_node.GetDisplayNode()
    if label_display:
        # Create custom color table for tumor (red)
        color_node = slicer.vtkMRMLColorTableNode()
        color_node.SetTypeToUser()
        color_node.SetName("TumorColorTable")
        color_node.SetNumberOfColors(2)
        color_node.SetColor(0, "Background", 0.0, 0.0, 0.0, 0.0)
        color_node.SetColor(1, "Tumor", 1.0, 0.0, 0.0, 1.0)  # Full opacity for tumor
        slicer.mrmlScene.AddNode(color_node)
        
        label_display.SetAndObserveColorNodeID(color_node.GetID())
        label_display.SetOpacity(LABEL_OPACITY)
        label_display.SetVisibility(True)
        # Enable outline mode for better visibility
        try:
            label_display.SetOutlineMode(1)  # Outline mode
        except:
            pass  # Some Slicer versions may not support this
    
    # Set slice viewer layers
    slicer.util.setSliceViewerLayers(
        background=ct_node,
        label=label_node,
        labelOpacity=LABEL_OPACITY
    )
    
    # Jump to best slice (found by regression model) and create bounding box ROI
    print(f"Jumping to best slice index: {BEST_SLICE_INDEX}")

    # Get slice views (use Red slice as reference)
    layout_manager = slicer.app.layoutManager()
    red_widget = layout_manager.sliceWidget("Red")
    red_logic = red_widget.sliceLogic()
    red_slice_node = red_logic.GetSliceNode()

    # Get CT volume as numpy array: shape (Z, Y, X)
    ct_array = slicer.util.arrayFromVolume(ct_node)
    num_slices = ct_array.shape[0]

    if BEST_SLICE_INDEX < num_slices:
        # --- 1) Jump to the correct axial slice using IJK->RAS ---
        # Build IJK->RAS matrix
        ijk_to_ras = vtk.vtkMatrix4x4()
        ct_node.GetIJKToRASMatrix(ijk_to_ras)

        # Use voxel (i=0, j=0, k=BEST_SLICE_INDEX) just to get its S (Z) position
        ijk = [0.0, 0.0, float(BEST_SLICE_INDEX), 1.0]
        ras = [0.0, 0.0, 0.0, 0.0]
        ijk_to_ras.MultiplyPoint(ijk, ras)

        # ras[2] is the S coordinate (slice offset)
        red_slice_node.SetSliceOffset(ras[2])
        red_logic.FitSliceToAll()
        print(f"Jumped to slice {BEST_SLICE_INDEX} at RAS Z={ras[2]:.2f}")

        # --- 2) Create a bounding-box ROI around the segmented tumor ---
        print("Creating bounding box ROI around tumor segmentation...")
        seg_array = slicer.util.arrayFromVolume(label_node)  # (Z, Y, X)
        seg_coords = np.argwhere(seg_array > 0)

        if seg_coords.size > 0:
            z_min, y_min, x_min = seg_coords.min(axis=0)
            z_max, y_max, x_max = seg_coords.max(axis=0)

            # Convert two opposite corners from IJK to RAS
            def ijk_to_ras_point(i, j, k):
                p_ijk = [float(i), float(j), float(k), 1.0]
                p_ras = [0.0, 0.0, 0.0, 0.0]
                ijk_to_ras.MultiplyPoint(p_ijk, p_ras)
                return p_ras[:3]

            corner1 = ijk_to_ras_point(x_min, y_min, z_min)
            corner2 = ijk_to_ras_point(x_max, y_max, z_max)

            center_ras = [
                0.5 * (corner1[0] + corner2[0]),
                0.5 * (corner1[1] + corner2[1]),
                0.5 * (corner1[2] + corner2[2]),
            ]
            radius_ras = [
                0.5 * abs(corner2[0] - corner1[0]),
                0.5 * abs(corner2[1] - corner1[1]),
                0.5 * abs(corner2[2] - corner1[2]),
            ]

            # --- 1) Mark the approximate centroid using Point List tool ---
            print("Creating centroid point (Point List)...")
            centroid_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Nodule_Centroid")
            centroid_node.AddControlPoint(center_ras)
            centroid_node.SetNthControlPointLabel(0, "Centroid")
            
            # Set centroid point display properties
            centroid_display = centroid_node.GetDisplayNode()
            if centroid_display:
                centroid_display.SetSelectedColor(1.0, 0.0, 0.0)  # Red
                centroid_display.SetOpacity(1.0)
                centroid_display.SetVisibility(True)
                centroid_display.SetGlyphType(centroid_display.Sphere3D)
                centroid_display.SetGlyphScale(5.0)  # Make it visible
                centroid_display.SetTextScale(4.0)
                centroid_display.SetPointLabelsVisibility(True)
            
            print(f"Created centroid point at {center_ras}")

            # --- 2) Create a bounding-box ROI around the segmented tumor ---
            roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "Nodule_ROI")
            roi_node.SetXYZ(*center_ras)
            roi_node.SetRadiusXYZ(*radius_ras)
            
            # Set ROI display properties for better visibility
            roi_display = roi_node.GetDisplayNode()
            if roi_display:
                roi_display.SetSelectedColor(1.0, 0.0, 0.0)  # Red
                roi_display.SetOpacity(0.5)
                roi_display.SetVisibility(True)
                roi_display.SetFillVisibility(True)
                roi_display.SetOutlineVisibility(True)
                # Note: SetOutlineThickness may not be available in all Slicer versions
                try:
                    roi_display.SetOutlineThickness(3.0)
                except AttributeError:
                    # Fallback: use line width if available
                    try:
                        roi_display.SetLineThickness(3.0)
                    except AttributeError:
                        pass  # Skip if not supported
            
            print(f"Created ROI bounding box at {center_ras} with radius {radius_ras}")
            
            # --- 3) Create contour from 2D mask on best slice ---
            print("Creating contour from segmentation mask...")
            slice_mask_2d = seg_array[BEST_SLICE_INDEX, :, :]
            if np.count_nonzero(slice_mask_2d) > 0:
                # Find contours using OpenCV
                import cv2
                binary_mask = (slice_mask_2d > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Use the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Create closed curve node for contour
                    contour_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode", "Nodule_Contour")
                    
                    # Convert contour points from IJK to RAS
                    for point in largest_contour:
                        y, x = point[0][1], point[0][0]  # OpenCV uses (x, y), but array is (y, x)
                        ijk_pt = [float(x), float(y), float(BEST_SLICE_INDEX), 1.0]
                        ras_pt = [0.0, 0.0, 0.0, 0.0]
                        ijk_to_ras.MultiplyPoint(ijk_pt, ras_pt)
                        contour_node.AddControlPoint(ras_pt[:3])
                    
                    # Set contour display properties
                    contour_display = contour_node.GetDisplayNode()
                    if contour_display:
                        contour_display.SetSelectedColor(1.0, 0.0, 0.0)  # Red
                        contour_display.SetOpacity(1.0)
                        contour_display.SetVisibility(True)
                        contour_display.SetLineThickness(3.0)
                        contour_display.SetFillVisibility(False)
                        contour_display.SetOutlineVisibility(True)
                    
                    print(f"Created contour with {len(largest_contour)} points")
            
            # Center all slice views on the nodule ROI so it is visible in Axial/Coronal/Sagittal views
            slicer.util.jumpSlices(*center_ras)
        else:
            print("Warning: segmentation is empty, cannot create ROI bounding box.")
    else:
        print(f"Warning: Slice index {BEST_SLICE_INDEX} out of range (0-{num_slices-1})")

    # Reset views
    slicer.util.resetSliceViews()
    
    print("=" * 60)
    print("Visualization complete!")
    print(f"Best slice (from regression): {BEST_SLICE_INDEX}")
    print("Tumor regions highlighted in red")
    print("")
    print("Appendix A Annotation Methods:")
    print("  1. ✓ Centroid point (Point List)")
    print("  2. ✓ ROI bounding box")
    print("  3. ✓ Contour (Closed Curve)")
    print("=" * 60)

if __name__ == "__main__":
    main()
