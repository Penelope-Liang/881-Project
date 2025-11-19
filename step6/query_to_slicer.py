import argparse, os, re, json, shutil
import numpy as np
import pandas as pd
import cv2
import torch
from pydicom import dcmread
import SimpleITK as sitk

from step4.models.unet import UNet


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


def match_rows(df: pd.DataFrame, operator: str, threshold: float):
	"""match the value with the operator and threshold

	args:
		df: input DataFrame
		operator: comparison operator
		threshold: threshold value

	returns:
		DataFrame: rows where pred_diameter_mm satisfies the condition
	"""

	diameter_column = df["pred_diameter_mm"]
	if operator == ">=": return df[diameter_column>=threshold]
	if operator == ">":  return df[diameter_column>threshold]
	if operator == "<=": return df[diameter_column<=threshold]
	if operator == "<":  return df[diameter_column<threshold]
	if operator == "=":  return df[np.isclose(diameter_column, threshold, atol=0.1)]
	raise ValueError(operator)


def save_nii_from_mask(mask2d: np.ndarray, spacing_xy: tuple, out_path: str):
	"""save the binary mask array as NIfTI file

	args:
		mask2d: binary mask array
		spacing_xy: spacing in (row, col)
		out_path: path to the output NIfTI file
	"""
	img = sitk.GetImageFromArray(mask2d.astype(np.uint8))
	# set spacing (x,y) -> (col,row) to SITK (x,y)
	img.SetSpacing((float(spacing_xy[1]), float(spacing_xy[0])))
	sitk.WriteImage(img, out_path)


def load_unet(checkpoint_path: str, device: torch.device):
	"""load_unet function loads the UNet model from the checkpoint file

	args:
		checkpoint_path: checkpoint file
		device: torch device

	returns:
		model: UNet model
	"""
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	model = UNet(in_channels=1, base=32).to(device)
	model.load_state_dict(checkpoint["model"])
	model.eval()
	return model


def normalize_img(arr):
	"""normalize_img function normalizes img array between 5th and 95th percentiles

	args:
		arr: img array

	returns:
		normalized img array with values in [0, 1]
	"""
	vmin, vmax = np.percentile(arr, [5,95])
	return np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)


def main():
	"""main function loads the pred_regression.csv and the query, selects the Top-K slices
	that satisfy the query condition based on pred_regression.csv,
	uses UNet model on those selected slices to generate segmentation masks, exports
	the corresponding dcm files and NIfTI masks to the output, and generates
	a topk_hits.csv file listing the selected results.
	"""
	parser = argparse.ArgumentParser(description="Step6: query -> TopK slices (Step5A) -> UNet mask -> export for 3D Slicer")
	parser.add_argument("--pred-csv", required=True, help="outputs/step5_reg/test_pred/pred_regression.csv")
	parser.add_argument("--query", required=True, help='e.g., ">=6mm"')
	parser.add_argument("--topk", type=int, default=10)
	parser.add_argument("--unet", required=True, help="outputs/step4/unet_best.path")
	parser.add_argument("--out", dest="out_dir", required=True)
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
	print("[step6] Device =", device)

	df = pd.read_csv(args.pred_csv)
	operator, threshold = parse_query(args.query)
	df_hit = match_rows(df, operator, threshold).copy()
	if df_hit.empty:
		print("[step6] no hits for", args.query)
		return
	df_hit = df_hit.sort_values("pred_diameter_mm", ascending=False).head(args.topk).reset_index(drop=True)

	os.makedirs(args.out_dir, exist_ok=True)
	df_hit.to_csv(os.path.join(args.out_dir, "topk_hits.csv"), index=False)

	unet = load_unet(args.unet, device)

	for i, row in df_hit.iterrows():
		dicom_path = row["dicom_path"]
		if not os.path.exists(dicom_path):
			print("[step6] Missing:", dicom_path); continue
		shutil.copy2(dicom_path, os.path.join(args.out_dir, f"hit_{i:03d}.dcm"))
		dcm_dataset = dcmread(dicom_path)
		img_arr = dcm_dataset.pixel_array.astype(np.float32)
		img_normalized = normalize_img(img_arr)
		img_resized = cv2.resize(img_normalized, (256,256), interpolation=cv2.INTER_LINEAR)
		input_tensor = torch.from_numpy(img_resized[None,None,...]).to(device)
		with torch.no_grad():
			prob = torch.sigmoid(unet(input_tensor))[0,0].detach().cpu().numpy()
		mask = (cv2.resize(prob, (img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
		spacing = (float(dcm_dataset.PixelSpacing[0]), float(dcm_dataset.PixelSpacing[1])) if hasattr(dcm_dataset, "PixelSpacing") else (1.0,1.0)
		out_nii = os.path.join(args.out_dir, f"hit_{i:03d}_mask.nii.gz")
		save_nii_from_mask(mask*255, spacing, out_nii)
		print("[step6] Wrote:", out_nii)

	print("[step6] Wrote hits CSV and DICOM + NIfTI to", args.out_dir)


if __name__ == "__main__":
	main()
