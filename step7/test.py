from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
import cv2
from typing import Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step4.models.unet import UNet
from step5.regression.models_reg import ResNetRegBin


# Slicer executable candidates
SLICER_CANDIDATES = [
    "/Applications/Slicer.app/Contents/MacOS/Slicer",
    "/Applications/3D Slicer.app/Contents/MacOS/Slicer",
    "/Applications/Slicer-5.6.app/Contents/MacOS/Slicer",
    "/Applications/Slicer-5.4.app/Contents/MacOS/Slicer",
    "/usr/bin/Slicer",
]


def normalize_img(arr: np.ndarray) -> np.ndarray:
    """Normalize image array between 5th and 95th percentiles"""
    vmin, vmax = np.percentile(arr, [5, 95])
    return np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)


def load_dicom_to_volume(dicom_dir: str) -> Tuple[sitk.Image, np.ndarray]:
    """Load DICOM series as 3D volume"""
    print("[Integrated Pipeline] Loading DICOM series...")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    if not dicom_names:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    
    print(f"[Integrated Pipeline] Found {len(dicom_names)} DICOM files")
    reader.SetFileNames(dicom_names)
    ct_volume = reader.Execute()
    
    ct_array = sitk.GetArrayFromImage(ct_volume)
    print(f"[Integrated Pipeline] Volume shape: {ct_array.shape}")
    
    return ct_volume, ct_array


def load_nifti_volume(nifti_path: str) -> Tuple[sitk.Image, np.ndarray]:
    """Load NIfTI volume"""
    print(f"[Integrated Pipeline] Loading NIfTI: {nifti_path}")
    ct_volume = sitk.ReadImage(nifti_path)
    ct_array = sitk.GetArrayFromImage(ct_volume)
    print(f"[Integrated Pipeline] Volume shape: {ct_array.shape}")
    return ct_volume, ct_array


def load_unet(checkpoint_path: str, device: torch.device):
    """Load UNet model from checkpoint"""
    print(f"[Integrated Pipeline] Loading UNet: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = UNet(in_channels=1, base=32).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_regression_model(checkpoint_path: str, device: torch.device):
    """Load regression model from checkpoint"""
    print(f"[Integrated Pipeline] Loading Regression: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ResNetRegBin(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def find_best_slice_with_regression(
    ct_array: np.ndarray,
    regression_model,
    device: torch.device
) -> Tuple[int, float]:
    """
    Use Step5 regression model to find the most relevant slice index
    
    Returns:
        Tuple of (best_slice_index, best_score)
    """
    print("[Integrated Pipeline] Finding best slice using regression model...")
    
    regression_scores = []
    num_slices = ct_array.shape[0]
    
    for slice_idx in range(num_slices):
        if slice_idx % 50 == 0:
            print(f"  Processing slice {slice_idx}/{num_slices}")
        
        slice_data = ct_array[slice_idx]
        
        # Normalize
        slice_norm = normalize_img(slice_data.astype(np.float32))
        
        # Resize to 256x256 for regression model
        slice_resized = cv2.resize(slice_norm, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        slice_tensor = torch.from_numpy(slice_resized[None, None, ...]).float().to(device)
        
        # Regression inference
        with torch.no_grad():
            pred_diameter, pred_bin = regression_model(slice_tensor)
            regression_scores.append(pred_diameter.item())
    
    # Find best slice (highest regression score = largest predicted diameter)
    best_slice_index = int(np.argmax(regression_scores))
    best_score = regression_scores[best_slice_index]
    
    print(f"[Integrated Pipeline] Best slice index: {best_slice_index}")
    print(f"[Integrated Pipeline] Best regression score (predicted diameter): {best_score:.2f} mm")
    
    return best_slice_index, best_score



def generate_segmentation_for_slice(
    ct_array: np.ndarray,
    slice_index: int,
    unet_model,
    device: torch.device
) -> np.ndarray:
    """
    Use Step4 UNet model to generate segmentation mask for a specific slice
    
    Returns:
        2D segmentation mask
    """
    print(f"[Integrated Pipeline] Generating segmentation for slice {slice_index}...")
    
    slice_data = ct_array[slice_index]
    
    # Normalize
    slice_norm = normalize_img(slice_data.astype(np.float32))
    
    # Resize to 256x256 for UNet
    slice_resized = cv2.resize(slice_norm, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    slice_tensor = torch.from_numpy(slice_resized[None, None, ...]).float().to(device)
    
    # UNet inference
    with torch.no_grad():
        seg_logits = unet_model(slice_tensor)
        seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
        
        # Resize back to original size
        seg_mask_full = cv2.resize(seg_prob, 
                                  (slice_data.shape[1], slice_data.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        seg_mask_binary = (seg_mask_full >= 0.5).astype(np.uint8)
    
    print(f"[Integrated Pipeline] Segmentation complete. Tumor pixels: {(seg_mask_binary > 0).sum()}")
    
    return seg_mask_binary


# Helper to load Step4 PNG mask
def load_step4_mask_from_png(step4_dir: str,
                             slice_index: int,
                             target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Load 2D mask PNG produced by Step4 for a given slice index and resize it to match the CT slice shape.

    Expected filename pattern in step4_dir:
        {slice_index:03d}_pred.png
    e.g.: 000_pred.png, 001_pred.png, ...
    """
    from pathlib import Path

    H, W = target_shape

    png_name = f"{slice_index:03d}_pred.png"
    png_path = Path(step4_dir) / png_name

    if not png_path.exists():
        raise FileNotFoundError(f"Step4 mask PNG not found: {png_path}")

    # Read as grayscale
    mask_png = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if mask_png is None:
        raise RuntimeError(f"Failed to read PNG: {png_path}")

    # Binarize: any non-zero pixel is considered tumor
    mask01 = (mask_png > 0).astype(np.uint8)

    # Resize to match (H, W) of CT slice
    if mask01.shape != (H, W):
        mask01 = cv2.resize(mask01, (W, H), interpolation=cv2.INTER_NEAREST)

    print(f"[Integrated Pipeline] Loaded Step4 PNG mask: {png_path}, tumor pixels: {(mask01 > 0).sum()}")
    return mask01


def create_3d_segmentation_volume(
    ct_array: np.ndarray,
    best_slice_index: int,
    slice_mask: np.ndarray
) -> np.ndarray:
    """Create 3D segmentation volume with mask only on the best slice"""
    segmentation_3d = np.zeros_like(ct_array, dtype=np.uint8)
    segmentation_3d[best_slice_index] = slice_mask
    return segmentation_3d


def create_slicer_script(
    ct_volume_path: str,
    segmentation_path: str,
    best_slice_index: int,
    output_dir: Path,
    window: float = 1500.0,
    level: float = -550.0,
    label_opacity: float = 0.9
) -> Path:
    """
    Create 3D Slicer automation script that:
    1. Loads CT volume
    2. Loads segmentation mask
    3. Jumps to the best slice index
    4. Highlights tumor regions
    """
    script_path = output_dir / "slicer_launcher.py"
    
    # Convert to absolute paths for Slicer script
    ct_volume_path_abs = os.path.abspath(ct_volume_path)
    segmentation_path_abs = os.path.abspath(segmentation_path)
    
    script_content = f'''#!/usr/bin/env python3
"""
3D Slicer Automation Script - Integrated Pipeline
Automatically loads CT, segmentation, and jumps to best slice
"""

import slicer
import numpy as np
import vtk
import os

# Configuration - using absolute paths
CT_VOLUME_PATH = r"{ct_volume_path_abs}"
SEGMENTATION_PATH = r"{segmentation_path_abs}"
BEST_SLICE_INDEX = {best_slice_index}
WINDOW = {window}
LEVEL = {level}
LABEL_OPACITY = {label_opacity}

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
        print("Applied window/level: Window={{}}, Level={{}}".format(WINDOW, LEVEL))
    
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
    print(f"Jumping to best slice index: {{BEST_SLICE_INDEX}}")

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
        print(f"Jumped to slice {{BEST_SLICE_INDEX}} at RAS Z={{ras[2]:.2f}}")

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
            
            print(f"Created centroid point at {{center_ras}}")

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
            
            print(f"Created ROI bounding box at {{center_ras}} with radius {{radius_ras}}")
            
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
                    
                    print(f"Created contour with {{len(largest_contour)}} points")
            
            # Center all slice views on the nodule ROI so it is visible in Axial/Coronal/Sagittal views
            slicer.util.jumpSlices(*center_ras)
        else:
            print("Warning: segmentation is empty, cannot create ROI bounding box.")
    else:
        print(f"Warning: Slice index {{BEST_SLICE_INDEX}} out of range (0-{{num_slices-1}})")

    # Reset views
    slicer.util.resetSliceViews()
    
    print("=" * 60)
    print("Visualization complete!")
    print(f"Best slice (from regression): {{BEST_SLICE_INDEX}}")
    print("Tumor regions highlighted in red")
    print("")
    print("Appendix A Annotation Methods:")
    print("  1. ✓ Centroid point (Point List)")
    print("  2. ✓ ROI bounding box")
    print("  3. ✓ Contour (Closed Curve)")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''
    
    script_path.write_text(script_content, encoding='utf-8')
    print(f"[Integrated Pipeline] Created Slicer script: {script_path}")
    
    return script_path


def find_slicer_executable(explicit: Optional[str] = None) -> Optional[str]:
    """Find 3D Slicer executable"""
    if explicit:
        explicit = os.path.abspath(explicit)
        return explicit if os.path.exists(explicit) else None
    
    for candidate in SLICER_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def launch_slicer(slicer_executable: str, script_path: Path) -> None:
    """Launch 3D Slicer with automation script"""
    print(f"[Integrated Pipeline] Launching Slicer: {slicer_executable}")
    print(f"[Integrated Pipeline] Automation script: {script_path}")
    subprocess.Popen([slicer_executable, "--python-script", str(script_path)])


def run_integrated_pipeline(
    input_path: str,
    unet_model_path: str,
    regression_model_path: str,
    output_dir: str,
    step4_dir: Optional[str] = None,
    slicer_executable: Optional[str] = None,
    no_launch: bool = False
) -> dict:
    """
    Run integrated pipeline according to assignment requirements:
    1. Load patient CT (DICOM or .nii.gz)
    2. Use Step5 regression to find most relevant slice index
    3. Use Step4 UNet to generate segmentation for that slice
    4. Create 3D segmentation volume
    5. Generate Slicer script and optionally launch Slicer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Integrated Pipeline] Using device: {device}")
    
    # Step 1: Load CT volume
    print("\n" + "=" * 60)
    print("Step 1: Load CT Volume")
    print("=" * 60)
    
    if os.path.isdir(input_path):
        # DICOM directory
        ct_volume, ct_array = load_dicom_to_volume(input_path)
        ct_volume_path = str(output_path / "ct_volume.nii.gz")
        sitk.WriteImage(ct_volume, ct_volume_path)
        print(f"[Integrated Pipeline] Saved CT volume: {ct_volume_path}")
    elif input_path.endswith(('.nii', '.nii.gz')):
        # NIfTI file
        ct_volume, ct_array = load_nifti_volume(input_path)
        ct_volume_path = input_path
    else:
        raise ValueError(f"Unsupported input format: {input_path}")
    
    # Step 2: Load models
    print("\n" + "=" * 60)
    print("Step 2: Load Models")
    print("=" * 60)
    
    # Regression model is always needed
    regression_model = load_regression_model(regression_model_path, device)
    
    # UNet is only needed if we are not using precomputed Step4 PNG masks
    unet_model = None
    if step4_dir is None:
        unet_model = load_unet(unet_model_path, device)
    else:
        print(f"[Integrated Pipeline] Using Step4 PNG masks from: {step4_dir} (will not run UNet)")
    
    # Step 3: Find best slice using regression
    print("\n" + "=" * 60)
    print("Step 3: Find Best Slice (Step5 Regression)")
    print("=" * 60)
    
    best_slice_index, best_score = find_best_slice_with_regression(
        ct_array, regression_model, device
    )
    
    # Step 4: Generate segmentation for best slice
    print("\n" + "=" * 60)
    print("Step 4: Generate Segmentation (Step4)")
    print("=" * 60)
    
    if step4_dir is not None:
        # Use Step4 PNG output directly
        slice_mask = load_step4_mask_from_png(
            step4_dir=step4_dir,
            slice_index=best_slice_index,
            target_shape=ct_array[best_slice_index].shape
        )
    else:
        # Fall back to running UNet if no PNG masks are provided
        slice_mask = generate_segmentation_for_slice(
            ct_array, best_slice_index, unet_model, device
        )
    
    # Step 5: Create 3D segmentation volume
    print("\n" + "=" * 60)
    print("Step 5: Create 3D Segmentation Volume")
    print("=" * 60)
    
    segmentation_3d = create_3d_segmentation_volume(
        ct_array, best_slice_index, slice_mask
    )
    
    # Save segmentation volume
    seg_sitk = sitk.GetImageFromArray(segmentation_3d)
    seg_sitk.SetSpacing(ct_volume.GetSpacing())
    seg_sitk.SetOrigin(ct_volume.GetOrigin())
    seg_sitk.SetDirection(ct_volume.GetDirection())
    
    segmentation_path = str(output_path / "segmentation_volume.nii.gz")
    sitk.WriteImage(seg_sitk, segmentation_path)
    print(f"[Integrated Pipeline] Saved segmentation: {segmentation_path}")
    
    # Step 6: Create Slicer script
    print("\n" + "=" * 60)
    print("Step 6: Create 3D Slicer Automation Script")
    print("=" * 60)
    
    slicer_script = create_slicer_script(
        ct_volume_path=ct_volume_path,
        segmentation_path=segmentation_path,
        best_slice_index=best_slice_index,
        output_dir=output_path
    )
    
    # Step 7: Launch Slicer (if requested)
    if not no_launch:
        print("\n" + "=" * 60)
        print("Step 7: Launch 3D Slicer")
        print("=" * 60)
        
        slicer_exe = find_slicer_executable(slicer_executable)
        if not slicer_exe:
            print("[Warning] 3D Slicer not found")
            print(f"Run manually: <Slicer> --python-script {slicer_script}")
        else:
            launch_slicer(slicer_exe, slicer_script)
    
    # Summary
    summary = {
        "ct_volume_path": ct_volume_path,
        "segmentation_path": segmentation_path,
        "best_slice_index": best_slice_index,
        "best_regression_score": best_score,
        "slicer_script": str(slicer_script),
        "tumor_pixels": int((slice_mask > 0).sum())
    }
    
    print("\n" + "=" * 60)
    print("Pipeline Complete - Summary")
    print("=" * 60)
    print(f"✓ CT Volume: {summary['ct_volume_path']}")
    print(f"✓ Segmentation: {summary['segmentation_path']}")
    print(f"✓ Best Slice Index (from regression): {summary['best_slice_index']}")
    print(f"✓ Best Regression Score: {summary['best_regression_score']:.2f} mm")
    print(f"✓ Tumor Pixels: {summary['tumor_pixels']}")
    print(f"✓ Slicer Script: {summary['slicer_script']}")
    print("=" * 60)
    
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step7 Integrated Pipeline - Assignment Requirements"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CT: DICOM directory or .nii.gz file"
    )
    parser.add_argument(
        "--unet",
        required=True,
        help="Path to Step4 UNet model checkpoint"
    )
    parser.add_argument(
        "--regression",
        required=True,
        help="Path to Step5 regression model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/step7_integrated",
        help="Output directory"
    )
    parser.add_argument(
        "--step4-dir",
        help="Directory containing Step4 PNG masks (e.g., 000_pred.png, 001_pred.png, ...)"
    )
    parser.add_argument(
        "--slicer",
        help="Path to Slicer executable (optional)"
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Do not automatically launch Slicer"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 60)
    print("Step7 Integrated Pipeline - Assignment Requirements")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"UNet model: {args.unet}")
    print(f"Regression model: {args.regression}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Check inputs
    if not os.path.exists(args.input):
        print(f"[Error] Input does not exist: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.unet):
        print(f"[Error] UNet model does not exist: {args.unet}")
        sys.exit(1)
    if not os.path.exists(args.regression):
        print(f"[Error] Regression model does not exist: {args.regression}")
        sys.exit(1)
    
    # Run pipeline
    try:
        run_integrated_pipeline(
            input_path=args.input,
            unet_model_path=args.unet,
            regression_model_path=args.regression,
            output_dir=args.output_dir,
            step4_dir=args.step4_dir,
            slicer_executable=args.slicer,
            no_launch=args.no_launch
        )
    except Exception as e:
        print(f"[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
