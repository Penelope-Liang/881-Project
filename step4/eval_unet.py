import argparse
import os
import ast
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from pydicom import dcmread
from PIL import Image
from skimage.transform import resize as sk_resize
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> None:
	os.makedirs(p, exist_ok=True)


def norm_img(arr: np.ndarray) -> np.ndarray:
	vmin, vmax = np.percentile(arr, [5, 95])
	arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
	return arr


def make_crop(img: np.ndarray, pts: List[List[int]], pad: int = 16, out_size: int = 256):
	xs = [p[0] for p in pts]
	ys = [p[1] for p in pts]
	xmin, ymin, xmax, ymax = min(xs)-pad, min(ys)-pad, max(xs)+pad, max(ys)+pad
	xmin = max(0, xmin)
	ymin = max(0, ymin)
	xmax = min(img.shape[1]-1, xmax)
	ymax = min(img.shape[0]-1, ymax)
	crop = img[ymin:ymax+1, xmin:xmax+1]
	crop_r = sk_resize(crop, (out_size, out_size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
	from PIL import ImageDraw
	poly = [(int(x), int(y)) for x, y in pts]
	m_img = Image.new('L', (img.shape[1], img.shape[0]), 0)
	ImageDraw.Draw(m_img).polygon(poly, outline=1, fill=1)
	msk = np.array(m_img, dtype=np.float32)
	msk = msk[ymin:ymax+1, xmin:xmax+1]
	msk_r = sk_resize(msk, (out_size, out_size), order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
	return crop_r, msk_r


def compute_metrics_bin(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6):
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


def _patient_from_path(path: str) -> Optional[str]:
	import re
	m = re.search(r"(LIDC-IDRI-\d{4})", path.replace('\\','/'))
	return m.group(1) if m else None


def _load_patient_set(pfile: Optional[str]) -> Optional[Set[str]]:
	if not pfile:
		return None
	s = set()
	with open(pfile, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line:
				s.add(line)
	return s


def main():
	parser = argparse.ArgumentParser(description='Step4: Evaluate UNet')
	parser.add_argument('--roi-csv', required=True)
	parser.add_argument('--model', required=True)
	parser.add_argument('--out', required=True)
	parser.add_argument('--img-size', type=int, default=256)
	parser.add_argument('--limit', type=int, default=1000)
	parser.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
	parser.add_argument('--patients-file', type=str, default=None, help='Restrict evaluation to these patients')
	args = parser.parse_args()

	ensure_dir(args.out)
	prev_dir = os.path.join(args.out, 'previews')
	ensure_dir(prev_dir)

	import torch
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if args.device == 'cpu':
		device = torch.device('cpu')
	elif args.device == 'cuda':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if device.type == 'cuda':
		try:
			cap = torch.cuda.get_device_capability(0)
			arch = f"sm_{cap[0]}{cap[1]}"
			compiled = torch.cuda.get_arch_list()
			if arch not in compiled:
				print(f"[eval] WARNING: GPU capability {arch} not in compiled arches {compiled}. Falling back to CPU.")
				device = torch.device('cpu')
		except Exception:
			pass
	print(f"[eval] device={device}")

	from step4.models.unet import UNet

	df = pd.read_csv(args.roi_csv)
	pset = _load_patient_set(args.patients_file)
	if pset:
		df = df[df['dicom_path'].astype(str).apply(lambda p: _patient_from_path(p) in pset)]
	if args.limit and args.limit > 0:
		df = df.sample(n=min(args.limit, len(df)), random_state=42)

	ckpt = torch.load(args.model, map_location='cpu')
	model = UNet(in_ch=1, base=32).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()

	per_rows = []
	ths = np.linspace(0.05, 0.95, 19)
	micro_counts = np.zeros((len(ths), 4), dtype=np.float64)

	for i, row in df.reset_index(drop=True).iterrows():
		ds = dcmread(row['dicom_path'])
		img = norm_img(ds.pixel_array.astype(np.float32))
		pts = ast.literal_eval(row['points_json']) if isinstance(row['points_json'], str) else row['points_json']
		crop, gt = make_crop(img, pts, pad=16, out_size=args.img_size)
		inp = torch.from_numpy(crop[None, None, ...].astype(np.float32)).to(device)
		with torch.no_grad():
			prob = torch.sigmoid(model(inp))[0, 0].detach().cpu().numpy()
		pred = (prob >= 0.5).astype(np.float32)
		dice, iou, prec, rec, (tp, fp, fn, tn) = compute_metrics_bin(pred, gt)
		per_rows.append({'dicom_path': row['dicom_path'], 'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec})
		for j, t in enumerate(ths):
			p = (prob >= t).astype(np.float32)
			gtb = gt.astype(np.float32)
			micro_counts[j, 0] += (p * gtb).sum()
			micro_counts[j, 1] += (p * (1 - gtb)).sum()
			micro_counts[j, 2] += ((1 - p) * gtb).sum()
			micro_counts[j, 3] += ((1 - p) * (1 - gtb)).sum()
		if i < 12:
			Image.fromarray((crop * 255).astype(np.uint8)).save(os.path.join(prev_dir, f'{i:03d}_img.png'))
			Image.fromarray((gt * 255).astype(np.uint8)).save(os.path.join(prev_dir, f'{i:03d}_gt.png'))
			Image.fromarray((prob * 255).astype(np.uint8)).save(os.path.join(prev_dir, f'{i:03d}_prob.png'))

	per_df = pd.DataFrame.from_records(per_rows)
	per_csv = os.path.join(args.out, 'metrics_per_sample.csv')
	per_df.to_csv(per_csv, index=False)

	summary = {
		'num_samples': int(len(per_df)),
		'mean_dice': float(per_df['dice'].mean()) if len(per_df) else 0.0,
		'mean_iou': float(per_df['iou'].mean()) if len(per_df) else 0.0,
		'mean_precision': float(per_df['precision'].mean()) if len(per_df) else 0.0,
		'mean_recall': float(per_df['recall'].mean()) if len(per_df) else 0.0,
	}
	import json
	with open(os.path.join(args.out, 'summary.json'), 'w', encoding='utf-8') as f:
		json.dump(summary, f, indent=2)

	prec_list, rec_list = [], []
	for j, t in enumerate(ths):
		tp, fp, fn, tn = micro_counts[j]
		precision = tp / (tp + fp + 1e-6)
		recall = tp / (tp + fn + 1e-6)
		prec_list.append(precision)
		rec_list.append(recall)
	pd.DataFrame({'threshold': ths, 'precision': prec_list, 'recall': rec_list}).to_csv(os.path.join(args.out, 'pr_curve.csv'), index=False)

	plt.figure(figsize=(5,4))
	plt.plot(rec_list, prec_list, marker='o')
	plt.xlabel('Recall (micro)')
	plt.ylabel('Precision (micro)')
	plt.title('PR Curve')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(args.out, 'pr_curve.png'))
	plt.close()

	plt.figure(figsize=(5,4))
	per_df['dice'].hist(bins=20)
	plt.xlabel('Dice')
	plt.ylabel('Count')
	plt.tight_layout()
	plt.savefig(os.path.join(args.out, 'dice_hist.png'))
	plt.close()

	print(f"[eval] device={device}")
	print(f"[eval] wrote summary: {os.path.join(args.out, 'summary.json')}")
	print(f"[eval] wrote per-sample: {per_csv}")
	print(f"[eval] wrote PR and hist in: {args.out}")
	print(f"[eval] previews: {prev_dir}")


if __name__ == '__main__':
	main()
