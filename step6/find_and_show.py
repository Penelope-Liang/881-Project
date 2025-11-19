import argparse, os, re, shutil, subprocess
import numpy as np
import pandas as pd
import torch
from pydicom import dcmread
import SimpleITK as sitk
from step4.models.unet import UNet

try:
	import cv2
except Exception:
	cv2 = None


def ensure_dir(path: str) -> None:
	"""ensure_dir function ensures directory exist

	args:
		path: directory path
	"""
	os.makedirs(path, exist_ok=True)


def parse_query(query: str):
	"""parse_query function parses query string into operator and threshold value

	args:
		query: query like ">=3mm"

	returns:
		tuple of operator, threshold
	"""
	query_normalized = query.strip().lower().replace(" ", "")
	match = re.match(r"(>=|<=|>|<|=)?([0-9]+\.?[0-9]*)mm", query_normalized)
	if not match:
		raise ValueError(f"Invalid query: {query}")
	operator = match.group(1) or ">="
	threshold = float(match.group(2))
	return operator, threshold


def match_value(value: float, operator: str, threshold: float) -> bool:
	"""check if value matches the operator and threshold condition

	args:
		value: value to check
		operator: comparison operator
		threshold: threshold value

	returns:
		True if value matches the condition, False otherwise
	"""
	if operator == ">=": return value >= threshold
	if operator == ">":  return value >  threshold
	if operator == "<=": return value <= threshold
	if operator == "<":  return value <  threshold
	if operator == "=":  return abs(value - threshold) <= 0.1
	raise ValueError(operator)


def normalize_img(arr):
	"""normalize_img function normalizes img array between 5th and 95th percentiles

	args:
		arr: img array

	returns:
		normalized img array with values in [0, 1]
	"""
	vmin, vmax = np.percentile(arr, [5,95])
	return np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)


def save_nii_from_mask(mask2d: np.ndarray, spacing_xy: tuple, out_path: str):
	"""save binary mask array as NIfTI file

	args:
		mask2d: binary mask array.
		spacing_xy: spacing in (row, col)
		out_path: path to output NIfTI file
	"""
	img = sitk.GetImageFromArray(mask2d.astype(np.uint8))
	img.SetSpacing((float(spacing_xy[1]), float(spacing_xy[0])))
	sitk.WriteImage(img, out_path)


def load_unet(checkpoint_path: str, device: torch.device):
	"""load UNet model from checkpoint file

	args:
		checkpoint_path: path to checkpoint file
		device: torch device either CPU or CUDA

	returns:
		load UNet model in evaluation mode
	"""
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	model = UNet(in_channels=1, base=32).to(device)
	model.load_state_dict(checkpoint["model"])
	model.eval()
	return model


def equivalent_diameter_mm(mask: np.ndarray, spacing_xy: tuple) -> float:
	"""calculate equivalent diameter of binary mask

	Args:
		mask: binary mask array.
		spacing_xy: pixel spacing in (row, col)

	Returns:
		equivalent diameter
	"""
	px_area = float(spacing_xy[0]) * float(spacing_xy[1])
	area_px = float((mask > 0).sum())
	area_mm2 = area_px * px_area
	if area_mm2 <= 0:
		return 0.0
	return float(np.sqrt(4.0 * area_mm2 / np.pi))


def process_series(dicom_path: str, unet, device: torch.device, operator: str, threshold: float):
	"""process dcm series with UNet and return segmentation results

	args:
		dicom_path: path to dcm file
		unet: UNet model for segmentation
		device: torch device (CPU or CUDA)
		operator: comparison operator
		threshold: threshold value

	returns:
		tuple of matches_threshold, diameter_mm, mask, spacing:
		- matches_threshold: whether diameter satisfies the query condition
		- diameter_mm: diameter
		- mask: binary segmentation mask array
		- spacing: spacing in (row, col)
	"""
	dcm_dataset = dcmread(dicom_path)
	img_arr = dcm_dataset.pixel_array.astype(np.float32)
	img_normalized = normalize_img(img_arr)
	# resize to 256 for UNet
	if cv2 is None:
		from skimage.transform import resize as sk_resize
		img_resized = sk_resize(img_normalized, (256,256), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
	else:
		img_resized = cv2.resize(img_normalized, (256,256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
	input_tensor = torch.from_numpy(img_resized[None,None,...]).to(device)
	with torch.no_grad():
		prob = torch.sigmoid(unet(input_tensor))[0,0].detach().cpu().numpy()
	# upsample back and binarize
	if cv2 is None:
		from skimage.transform import resize as sk_resize
		prob_full = sk_resize(prob, (img_arr.shape[0], img_arr.shape[1]), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
	else:
		prob_full = cv2.resize(prob, (img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
	mask = (prob_full >= 0.5).astype(np.uint8)
	# spacing
	spacing = (float(getattr(dcm_dataset, "PixelSpacing", [1.0,1.0])[0]), float(getattr(dcm_dataset, "PixelSpacing", [1.0,1.0])[1]))
	diameter_mm = equivalent_diameter_mm(mask, spacing)
	matches_threshold = match_value(diameter_mm, operator, threshold)
	return matches_threshold, diameter_mm, mask, spacing


def main():
	"""process dcm folder, generate segmentations, select Top-K slices, and export results, 
	scans a dcm folder, uses UNet to generate segmentation masks for each slice,
	computes diameters from masks, selects Top-K slices matching the query condition,
	exports dcm files and NIfTI masks, generates topk_hits.csv, and optionally
	launches 3D Slicer to view the output.
	"""
	parser = argparse.ArgumentParser(description="Step6: run on a dcm folder (no CSV), select Top‑K by UNet-estimated size, export to Slicer")
	parser.add_argument("--data-root", required=True)
	parser.add_argument("--query", required=True, help='e.g., ">=6mm"')
	parser.add_argument("--topk", type=int, default=10)
	parser.add_argument("--unet", required=True)
	parser.add_argument("--out", dest="out_dir", required=True)
	parser.add_argument("--slicer", default=None, help="Path to Slicer executable to auto-open (optional)")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		try:
			cap = torch.cuda.get_device_capability(0)
			arch = f"sm_{cap[0]}{cap[1]}"
			compiled = torch.cuda.get_arch_list()
			if arch not in compiled:
				print(f"[step6] WARN: GPU {arch} not in {compiled}, fallback CPU")
				device = torch.device("cpu")
		except Exception:
			pass
	print("[step6] device =", device)

	operator, threshold = parse_query(args.query)
	unet = load_unet(args.unet, device)

	dcm_files = []
	for r, _, files in os.walk(args.data_root):
		for n in files:
			if n.lower().endswith(".dcm"):
				dcm_files.append(os.path.join(r,n))
	if not dcm_files:
		print("[step6] no dicom files under", args.data_root); return

	rows = []
	for p in dcm_files:
		try:
			matches_threshold, diameter_mm, mask, spacing = process_series(p, unet, device, operator, threshold)
			rows.append({"dicom_path": p, "pred_diameter_mm": diameter_mm, "matches_threshold": matches_threshold, "mask": mask, "spacing": spacing})
		except Exception as e:
			print("[step6] fail:", p, e)

	ensure_dir(args.out_dir)
	df = pd.DataFrame([{k:v for k,v in r.items() if k not in ("mask","spacing")} for r in rows])
	df_hit = df[df["matches_threshold"]].sort_values("pred_diameter_mm", ascending=False).head(args.topk).reset_index(drop=True)
	df_hit.to_csv(os.path.join(args.out_dir, "topk_hits.csv"), index=False)
	print("[step6] hits:", len(df_hit))

	for i, (_, rec) in enumerate(df_hit.iterrows()):
		p = rec["dicom_path"]
		shutil.copy2(p, os.path.join(args.out_dir, f"hit_{i:03d}.dcm"))
		mask = next(r["mask"] for r in rows if r["dicom_path"]==p)
		spacing = next(r["spacing"] for r in rows if r["dicom_path"]==p)
		out_nii = os.path.join(args.out_dir, f"hit_{i:03d}_mask.nii.gz")
		save_nii_from_mask(mask*255, spacing, out_nii)
		print("[step6] wrote:", out_nii)

	if args.slicer and os.path.exists(args.slicer):
		try:
			print("[step6] launching 3D Slicer...")
			subprocess.Popen([args.slicer, args.out_dir])
		except Exception as e:
			print("[step6] slicer launch failed:", e)

	print("[step6] done ->", args.out_dir)


if __name__ == "__main__":
	main()
