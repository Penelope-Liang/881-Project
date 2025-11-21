import os
import sys
import argparse
import json
import torch
import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def load_ct_volume(ct_dir):
    """
    load 3d ct volume from dicom directory
    
    args:
        ct_dir: directory containing dicom files
    
    returns:
        volume: 3d numpy array (z, h, w)
        dicom_files: list of dicom file paths sorted by slice location
    """
    print(f"Loading CT volume from: {ct_dir}")
    
    # read all dicom files
    dicom_files = []
    for root, dirs, files in os.walk(ct_dir):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {ct_dir}")
    
    # sort by slice location
    slices = []
    for dcm_path in dicom_files:
        dcm = pydicom.dcmread(dcm_path)
        slices.append((dcm, dcm_path))
    
    slices.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))
    
    # build volume
    volume = []
    sorted_paths = []
    for dcm, dcm_path in slices:
        pixel_array = dcm.pixel_array.astype(float)
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        volume.append(pixel_array)
        sorted_paths.append(dcm_path)
    
    volume = np.stack(volume, axis=0)
    print(f"  Volume shape: {volume.shape}")
    
    return volume, sorted_paths


def detect_nodules_yolo(volume, yolo_model, conf_threshold=0.25):
    """
    detect nodules on each slice using yolov8
    
    args:
        volume: 3d ct volume (z, h, w)
        yolo_model: trained yolov8 model
        conf_threshold: confidence threshold
    
    returns:
        detections: list of detections per slice
    """
    print("\nDetecting nodules with YOLOv8")
    
    detections = []
    for slice_idx in tqdm(range(volume.shape[0]), desc="Detecting"):
        slice_data = volume[slice_idx]
        
        # normalize to [0, 255]
        slice_norm = np.clip((slice_data + 1000) / 2000 * 255, 0, 255).astype(np.uint8)
        
        # convert to 3-channel for yolov8
        slice_rgb = np.stack([slice_norm] * 3, axis=-1)
        
        # run detection
        results = yolo_model(slice_rgb, conf=conf_threshold, verbose=False)
        
        slice_detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                slice_detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf)
                })
        
        detections.append(slice_detections)
    
    total_detections = sum(len(d) for d in detections)
    print(f"  Total detections: {total_detections}")
    
    return detections


def predict_diameters(volume, detections, reg_model, device):
    """
    predict nodule diameters using regression model
    
    args:
        volume: 3d ct volume (z, h, w)
        detections: list of detections per slice
        reg_model: trained regression model
        device: torch device
    
    returns:
        detections: updated detections with diameter predictions
    """
    print("\nPredicting nodule diameters")
    
    reg_model.eval()
    
    for slice_idx, slice_dets in enumerate(tqdm(detections, desc="Predicting")):
        if not slice_dets:
            continue
        
        slice_data = volume[slice_idx]
        
        for det in slice_dets:
            x1, y1, x2, y2 = det["bbox"]
            
            # extract roi
            x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
            x2_int, y2_int = min(slice_data.shape[1], int(x2)), min(slice_data.shape[0], int(y2))
            
            roi = slice_data[y1_int:y2_int, x1_int:x2_int]
            
            if roi.size == 0:
                det["diameter_mm"] = 0.0
                continue
            
            # resize to 64x64
            from skimage.transform import resize
            roi_resized = resize(roi, (64, 64), mode="constant", anti_aliasing=True)
            
            # normalize
            roi_norm = (roi_resized - roi_resized.mean()) / (roi_resized.std() + 1e-8)
            
            # predict
            with torch.no_grad():
                roi_tensor = torch.from_numpy(roi_norm).float().unsqueeze(0).unsqueeze(0).to(device)
                pred_diam, _ = reg_model(roi_tensor)  # returns (diameter, logits)
                pred_diam = pred_diam.item()
            
            det["diameter_mm"] = max(0.0, pred_diam)
    
    return detections


def filter_by_query(detections, query):
    """
    filter detections by keyword query
    
    args:
        detections: list of detections per slice
        query: query string (e.g., ">3mm", "<5mm", "3-5mm")
    
    returns:
        filtered_detections: filtered detections
    """
    print(f"\nFiltering by query: \"{query}\"")
    
    # parse query
    query = query.lower().replace(" ", "")
    
    def matches_query(diameter_mm):
        if ">" in query:
            threshold = float(query.replace(">", "").replace("mm", ""))
            return diameter_mm > threshold
        elif "<" in query:
            threshold = float(query.replace("<", "").replace("mm", ""))
            return diameter_mm < threshold
        elif "-" in query:
            parts = query.replace("mm", "").split("-")
            diameter_min, diameter_max = float(parts[0]), float(parts[1])
            return diameter_min <= diameter_mm <= diameter_max
        else:
            # exact match
            threshold = float(query.replace("mm", ""))
            return abs(diameter_mm - threshold) < 0.5
    
    filtered_detections = []
    for slice_dets in detections:
        filtered_slice_dets = [d for d in slice_dets if matches_query(d["diameter_mm"])]
        filtered_detections.append(filtered_slice_dets)
    
    total_filtered = sum(len(d) for d in filtered_detections)
    print(f"  Matching detections: {total_filtered}")
    
    return filtered_detections


def select_top1_slice(detections):
    """
    select top-1 most relevant slice (most detections, highest confidence)
    
    args:
        detections: filtered detections per slice
    
    returns:
        top1_idx: index of top-1 slice
        top1_detections: detections on top-1 slice
    """
    print("\nSelecting top-1 slice")
    
    scores = []
    for slice_idx, slice_dets in enumerate(detections):
        if not slice_dets:
            scores.append(0.0)
        else:
            # score = number of detections * average confidence
            avg_conf = np.mean([d["confidence"] for d in slice_dets])
            score = len(slice_dets) * avg_conf
            scores.append(score)
    
    top1_idx = int(np.argmax(scores))
    top1_detections = detections[top1_idx]
    
    print(f"Top-1 slice: {top1_idx}")
    print(f"Nodules: {len(top1_detections)}")
    
    return top1_idx, top1_detections


def segment_nodules_unet(volume, top1_idx, top1_detections, unet_model, device):
    """
    segment nodules on top-1 slice using unet
    
    args:
        volume: 3d ct volume (z, h, w)
        top1_idx: index of top-1 slice
        top1_detections: detections on top-1 slice
        unet_model: trained unet model
        device: torch device
    
    returns:
        segmentation: binary segmentation mask (h, w)
    """
    print("\nSegmenting nodules with UNet")
    
    unet_model.eval()
    
    slice_data = volume[top1_idx]
    H, W = slice_data.shape
    
    # initialize full-slice segmentation
    segmentation = np.zeros((H, W), dtype=np.uint8)
    
    for det in top1_detections:
        x1, y1, x2, y2 = det["bbox"]
        x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
        x2_int, y2_int = min(W, int(x2)), min(H, int(y2))
        
        # extract roi
        roi = slice_data[y1_int:y2_int, x1_int:x2_int]
        
        if roi.size == 0:
            continue
        
        # resize to 128x128
        from skimage.transform import resize
        roi_resized = resize(roi, (128, 128), mode="constant", anti_aliasing=True)
        
        # normalize
        roi_norm = (roi_resized - roi_resized.mean()) / (roi_resized.std() + 1e-8)
        
        # segment
        with torch.no_grad():
            roi_tensor = torch.from_numpy(roi_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            pred_mask = unet_model(roi_tensor)
            pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
        
        # resize back to original roi size
        pred_mask_resized = resize(pred_mask, (y2_int - y1_int, x2_int - x1_int), 
                                   mode="constant", anti_aliasing=False, order=0)
        pred_mask_resized = (pred_mask_resized > 0.5).astype(np.uint8)
        
        # place in full segmentation
        segmentation[y1_int:y2_int, x1_int:x2_int] = np.maximum(
            segmentation[y1_int:y2_int, x1_int:x2_int],
            pred_mask_resized
        )
    
    print(f"Segmentation complete")
    
    return segmentation


def export_for_slicer(volume, dicom_files, top1_idx, top1_detections, segmentation, output_dir):
    """
    export results for 3d slicer visualization
    
    args:
        volume: 3d ct volume (z, h, w)
        dicom_files: list of dicom file paths
        top1_idx: index of top-1 slice
        top1_detections: detections on top-1 slice
        segmentation: binary segmentation mask (h, w)
        output_dir: output directory
    """
    print(f"\nExporting results to: {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # export 3d volume as nifti
    print("  Exporting CT volume")
    volume_sitk = sitk.GetImageFromArray(volume)
    
    # get spacing from first dicom
    dcm = pydicom.dcmread(dicom_files[0])
    pixel_spacing = dcm.PixelSpacing
    slice_thickness = dcm.SliceThickness if hasattr(dcm, "SliceThickness") else 1.0
    volume_sitk.SetSpacing([float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)])
    
    sitk.WriteImage(volume_sitk, str(output_dir / "series.nii.gz"))
    
    # export 3d label volume
    print("Exporting segmentation")
    label_volume = np.zeros_like(volume, dtype=np.uint8)
    label_volume[top1_idx] = segmentation
    
    label_sitk = sitk.GetImageFromArray(label_volume)
    label_sitk.SetSpacing(volume_sitk.GetSpacing())
    sitk.WriteImage(label_sitk, str(output_dir / "series_label.nii.gz"))
    
    # export summary json
    print("Exporting summary")
    summary = {
        "top1_slice_index": int(top1_idx),
        "nodules": []
    }
    
    for det in top1_detections:
        summary["nodules"].append({
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "diameter_mm": det["diameter_mm"]
        })
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Export complete")


def run_pipeline(ct_dir, query, detector_path, unet_path, regressor_path, output_dir, conf_threshold=0.25, device="cuda"):
    """
    run the complete pipeline
    
    args:
        ct_dir: directory containing dicom files
        query: query string (e.g., ">3mm")
        detector_path: path to trained yolov8 model
        unet_path: path to trained unet model
        regressor_path: path to trained regression model
        output_dir: output directory for results
        conf_threshold: yolov8 confidence threshold
        device: torch device
    """
    
    print("End-to-End Nodule Detection and Segmentation Pipeline")
    
    # load models
    print("\nLoading models")
    yolo_model = YOLO(detector_path)
    
    from step4.models.unet import UNet
    unet_model = UNet(in_channels=1).to(device)
    unet_checkpoint = torch.load(unet_path, map_location=device)
    if isinstance(unet_checkpoint, dict) and "model" in unet_checkpoint:
        unet_model.load_state_dict(unet_checkpoint["model"])
    else:
        unet_model.load_state_dict(unet_checkpoint)
    
    from step5.regression.models_reg import ResNetRegBin
    reg_model = ResNetRegBin().to(device)
    reg_checkpoint = torch.load(regressor_path, map_location=device)
    if isinstance(reg_checkpoint, dict) and "model" in reg_checkpoint:
        reg_model.load_state_dict(reg_checkpoint["model"])
    else:
        reg_model.load_state_dict(reg_checkpoint)
    
    print("Models loaded")
    
    # run pipeline
    volume, dicom_files = load_ct_volume(ct_dir)
    detections = detect_nodules_yolo(volume, yolo_model, conf_threshold)
    detections = predict_diameters(volume, detections, reg_model, device)
    filtered_detections = filter_by_query(detections, query)
    top1_idx, top1_detections = select_top1_slice(filtered_detections)
    segmentation = segment_nodules_unet(volume, top1_idx, top1_detections, unet_model, device)
    export_for_slicer(volume, dicom_files, top1_idx, top1_detections, segmentation, output_dir)
    
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print(f"Results saved to: {output_dir}")
    print("\nNext step: Visualize in 3D Slicer")
    print(f"  python step7/visualize.py --result-dir {output_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end nodule detection pipeline")
    parser.add_argument("--ct-dir", type=str, required=True, help="Directory containing DICOM files")
    parser.add_argument("--query", type=str, required=True, help="Query string (e.g., \">3mm\", \"<5mm\", \"3-5mm\")")
    parser.add_argument("--detector", type=str, required=True, help="Path to trained YOLOv8 model (.pt file)")
    parser.add_argument("--unet", type=str, required=True, help="Path to trained UNet model (.pth file)")
    parser.add_argument("--regressor", type=str, required=True, help="Path to trained regression model (.pth file)")
    parser.add_argument("--output-dir", type=str, default="plan_b/outputs/result", help="Output directory for results")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="YOLOv8 confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    run_pipeline(
        ct_dir=args.ct_dir,
        query=args.query,
        detector_path=args.detector,
        unet_path=args.unet,
        regressor_path=args.regressor,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        device=args.device
    )

