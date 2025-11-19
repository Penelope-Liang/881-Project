import argparse
import os
import ast
import numpy as np
import pandas as pd
from pydicom import dcmread
from PIL import Image
from skimage.transform import resize as sk_resize
import matplotlib.pyplot as plt
import re
from step4.models.unet import UNet
from typing import List, Optional, Set

def ensure_dir(path: str) -> None:
	"""ensure_dir function ensures directory exist

	args:
		path: directory path
	"""
	os.makedirs(path, exist_ok=True)


def normalize_img(arr: np.ndarray) -> np.ndarray:
	"""normalize_img function normalizes img array between 5th and 95th percentiles

	args:
		arr: img array

	returns:
		normalized img array with values in [0, 1]
	"""
	vmin, vmax = np.percentile(arr, [5, 95])
	arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
	return arr


def make_crop(img: np.ndarray, points: List[List[int]], pad: int = 16, out_size: int = 256):
	"""make_crop function crops img and create mask from ROI points, then resize both

	args:
		img: input img array
		points: list of (x, y) for ROI
		pad: padding pixels to add around ROI bounding box
		out_size: target size for resizing crop and mask

	returns:
		tuple of crop_resized, mask_resized
	"""
	xs = [p[0] for p in points]
	ys = [p[1] for p in points]
	xmin, ymin, xmax, ymax = min(xs)-pad, min(ys)-pad, max(xs)+pad, max(ys)+pad
	xmin = max(0, xmin)
	ymin = max(0, ymin)
	xmax = min(img.shape[1]-1, xmax)
	ymax = min(img.shape[0]-1, ymax)
	crop = img[ymin:ymax+1, xmin:xmax+1]
	crop_resized = sk_resize(crop, (out_size, out_size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
	from PIL import ImageDraw
	poly = [(int(x), int(y)) for x, y in points]
	mask_img = Image.new("L", (img.shape[1], img.shape[0]), 0)
	ImageDraw.Draw(mask_img).polygon(poly, outline=1, fill=1)
	mask = np.array(mask_img, dtype=np.float32)
	mask = mask[ymin:ymax+1, xmin:xmax+1]
	mask_resized = sk_resize(mask, (out_size, out_size), order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
	return crop_resized, mask_resized


def compute_metrics_bin(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6):
	"""compute binary segmentation metrics

	args:
		pred: predicted binary mask array
		gt: ground truth binary mask array
		eps: small epsilon value to avoid division by zero

	returns:
		tuple of dice, iou, precision, recall, (tp, fp, fn, tn)
	"""
	tp = float((pred * gt).sum())
	fp = float((pred * (1 - gt)).sum())
	fn = float(((1 - pred) * gt).sum())
	tn = float(((1 - pred) * (1 - gt)).sum())
	precision = tp / (tp + fp + eps)
	recall = tp / (tp + fn + eps)
	intersection = float((pred * gt).sum())
	union = float(((pred + gt) > 0).sum())
	iou = intersection / (union + eps)
	dice = 2 * intersection / (pred.sum() + gt.sum() + eps)
	return dice, iou, precision, recall, (tp, fp, fn, tn)


def patient_from_path(path: str) -> Optional[str]:
	"""patient_from_path function extracts patient id from file path

	args:
		path: file path containing patient id

	returns:
		patient id str like "LIDC-IDRI-0001" or None
	"""
	match = re.search(r"(LIDC-IDRI-\d{4})", path.replace("\\","/"))
	return match.group(1) if match else None


def load_patient_set(patient_file: Optional[str]) -> Optional[Set[str]]:
	"""load_patient_set function loads patient ids from file into a set

	args:
		patient_file: path to file containing patient ids

	returns:
		set of patient id str, or None
	"""
	if not patient_file:
		return None
	patient_set = set()
	with open(patient_file, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				patient_set.add(line)
	return patient_set


def main():
	"""main function evaluates UNet model on ROI dataset,loads ROI CSV, filters by patient set if provided, 
	loads UNet model, evaluates on samples, computes metrics including dice, iou, precision, recall,
	generates PR curve and dice histogram, saves results to output directory.
	"""
	parser = argparse.ArgumentParser(description="Step4: Evaluate UNet")
	parser.add_argument("--roi-csv", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--out", dest="out_dir", required=True)
	parser.add_argument("--img-size", type=int, default=256)
	parser.add_argument("--limit", type=int, default=1000)
	parser.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
	parser.add_argument("--patients-file", type=str, default=None, help="Restrict evaluation to these patients")
	args = parser.parse_args()

	ensure_dir(args.out_dir)
	preview_dir = os.path.join(args.out_dir, "previews")
	ensure_dir(preview_dir)

	import torch
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if args.device == "cpu":
		device = torch.device("cpu")
	elif args.device == "cuda":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		try:
			cap = torch.cuda.get_device_capability(0)
			arch = f"sm_{cap[0]}{cap[1]}"
			compiled = torch.cuda.get_arch_list()
			if arch not in compiled:
				print(f"[eval] WARNING: GPU capability {arch} not in compiled arches {compiled}. Falling back to CPU.")
				device = torch.device("cpu")
		except Exception:
			pass
	print(f"[eval] device={device}")

	df = pd.read_csv(args.roi_csv)
	patient_set = load_patient_set(args.patients_file)
	if patient_set:
		df = df[df["dicom_path"].astype(str).apply(lambda p: patient_from_path(p) in patient_set)]
	if args.limit and args.limit > 0:
		df = df.sample(n=min(args.limit, len(df)), random_state=42)

	checkpoint = torch.load(args.model, map_location="cpu")
	model = UNet(in_channels=1, base=32).to(device)
	model.load_state_dict(checkpoint["model"])
	model.eval()

	per_sample_rows = []
	thresholds = np.linspace(0.05, 0.95, 19)
	micro_counts = np.zeros((len(thresholds), 4), dtype=np.float64)

	for i, row in df.reset_index(drop=True).iterrows():
		dcm_dataset = dcmread(row["dicom_path"])
		img = normalize_img(dcm_dataset.pixel_array.astype(np.float32))
		points = ast.literal_eval(row["points_json"]) if isinstance(row["points_json"], str) else row["points_json"]
		crop, gt = make_crop(img, points, pad=16, out_size=args.img_size)
		input_tensor = torch.from_numpy(crop[None, None, ...].astype(np.float32)).to(device)
		with torch.no_grad():
			prob = torch.sigmoid(model(input_tensor))[0, 0].detach().cpu().numpy()
		pred = (prob >= 0.5).astype(np.float32)
		dice, iou, precision, recall, (tp, fp, fn, tn) = compute_metrics_bin(pred, gt)
		per_sample_rows.append({"dicom_path": row["dicom_path"], "dice": dice, "iou": iou, "precision": precision, "recall": recall})
		for j, t in enumerate(thresholds):
			p = (prob >= t).astype(np.float32)
			ground_truth_binary = gt.astype(np.float32)
			micro_counts[j, 0] += (p * ground_truth_binary).sum()
			micro_counts[j, 1] += (p * (1 - ground_truth_binary)).sum()
			micro_counts[j, 2] += ((1 - p) * ground_truth_binary).sum()
			micro_counts[j, 3] += ((1 - p) * (1 - ground_truth_binary)).sum()
		if i < 12:
			Image.fromarray((crop * 255).astype(np.uint8)).save(os.path.join(preview_dir, f"{i:03d}_img.png"))
			Image.fromarray((gt * 255).astype(np.uint8)).save(os.path.join(preview_dir, f"{i:03d}_gt.png"))
			Image.fromarray((prob * 255).astype(np.uint8)).save(os.path.join(preview_dir, f"{i:03d}_prob.png"))

	per_sample_df = pd.DataFrame.from_records(per_sample_rows)
	per_sample_csv = os.path.join(args.out_dir, "metrics_per_sample.csv")
	per_sample_df.to_csv(per_sample_csv, index=False)

	summary = {
		"num_samples": int(len(per_sample_df)),
		"mean_dice": float(per_sample_df["dice"].mean()) if len(per_sample_df) else 0.0,
		"mean_iou": float(per_sample_df["iou"].mean()) if len(per_sample_df) else 0.0,
		"mean_precision": float(per_sample_df["precision"].mean()) if len(per_sample_df) else 0.0,
		"mean_recall": float(per_sample_df["recall"].mean()) if len(per_sample_df) else 0.0,
	}
	import json
	with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	precision_list, recall_list = [], []
	for j, t in enumerate(thresholds):
		tp, fp, fn, tn = micro_counts[j]
		precision = tp / (tp + fp + 1e-6)
		recall = tp / (tp + fn + 1e-6)
		precision_list.append(precision)
		recall_list.append(recall)
	pd.DataFrame({"threshold": thresholds, "precision": precision_list, "recall": recall_list}).to_csv(os.path.join(args.out_dir, "pr_curve.csv"), index=False)

	plt.figure(figsize=(5,4))
	plt.plot(recall_list, precision_list, marker="o")
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title("Precision Recall Curve")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(args.out_dir, "precision_recall_curve.png"))
	plt.close()

	plt.figure(figsize=(5,4))
	per_sample_df["dice"].hist(bins=20)
	plt.xlabel("Dice")
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(os.path.join(args.out_dir, "dice_hist.png"))
	plt.close()

	print(f"[eval] device={device}")
	print(f"[eval] wrote summary: {os.path.join(args.out_dir, "summary.json")}")
	print(f"[eval] wrote per-sample: {per_sample_csv}")
	print(f"[eval] wrote PR and hist in: {args.out_dir}")
	print(f"[eval] previews: {preview_dir}")


if __name__ == "__main__":
	main()
