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


def ensure_dir(p: str):
	os.makedirs(p, exist_ok=True)


def parse_query(q: str):
	s = q.strip().lower().replace(' ', '')
	m = re.match(r'(>=|<=|>|<|=)?([0-9]+\.?[0-9]*)mm', s)
	if not m:
		raise ValueError(f"Invalid query: {q}")
	op = m.group(1) or '>='
	val = float(m.group(2))
	return op, val


def match_value(v: float, op: str, thr: float) -> bool:
	if op == '>=': return v >= thr
	if op == '>':  return v >  thr
	if op == '<=': return v <= thr
	if op == '<':  return v <  thr
	if op == '=':  return abs(v - thr) <= 0.1
	raise ValueError(op)


def norm_img(arr):
	vmin, vmax = np.percentile(arr, [5,95])
	return np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)


def save_nii_from_mask(mask2d: np.ndarray, spacing_xy: tuple, out_path: str):
	img = sitk.GetImageFromArray(mask2d.astype(np.uint8))
	img.SetSpacing((float(spacing_xy[1]), float(spacing_xy[0])))
	sitk.WriteImage(img, out_path)


def load_unet(ckpt_p: str, device: torch.device):
	ckpt = torch.load(ckpt_p, map_location='cpu')
	model = UNet(in_ch=1, base=32).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	return model


def equivalent_diameter_mm(mask: np.ndarray, spacing_xy: tuple) -> float:
	# area (mm^2) from pixel count * pixel area
	px_area = float(spacing_xy[0]) * float(spacing_xy[1])
	area_px = float((mask>0).sum())
	area_mm2 = area_px * px_area
	if area_mm2 <= 0:
		return 0.0
	return float(np.sqrt(4.0 * area_mm2 / np.pi))


def process_series(dcm_path: str, unet, device: torch.device, op: str, thr: float):
	ds = dcmread(dcm_path)
	px = ds.pixel_array.astype(np.float32)
	nm = norm_img(px)
	# resize to 256 for UNet
	if cv2 is None:
		from skimage.transform import resize as sk_resize
		img_r = sk_resize(nm, (256,256), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
	else:
		img_r = cv2.resize(nm, (256,256), interpolation=cv2.INTER_LINEAR)
	inp = torch.from_numpy(img_r[None,None,...]).to(device)
	with torch.no_grad():
		prob = torch.sigmoid(unet(inp))[0,0].detach().cpu().numpy()
	# upsample back and binarize
	if cv2 is None:
		from skimage.transform import resize as sk_resize
		prob_full = sk_resize(prob, (px.shape[0], px.shape[1]), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
	else:
		prob_full = cv2.resize(prob, (px.shape[1], px.shape[0]), interpolation=cv2.INTER_LINEAR)
	mask = (prob_full >= 0.5).astype(np.uint8)
	# spacing
	spacing = (float(getattr(ds, 'PixelSpacing', [1.0,1.0])[0]), float(getattr(ds, 'PixelSpacing', [1.0,1.0])[1]))
	diam_mm = equivalent_diameter_mm(mask, spacing)
	ok = match_value(diam_mm, op, thr)
	return ok, diam_mm, mask, spacing


def main():
	ap = argparse.ArgumentParser(description='Step6: run on a DICOM folder (no CSV), select Top‑K by UNet-estimated size, export to Slicer')
	ap.add_argument('--dicom-root', required=True)
	ap.add_argument('--query', required=True, help='e.g., ">=6mm"')
	ap.add_argument('--topk', type=int, default=10)
	ap.add_argument('--unet', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--slicer', default=None, help='Path to Slicer executable to auto-open (optional)')
	args = ap.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if device.type == 'cuda':
		try:
			cap = torch.cuda.get_device_capability(0)
			arch = f"sm_{cap[0]}{cap[1]}"
			compiled = torch.cuda.get_arch_list()
			if arch not in compiled:
				print(f"[step6] WARN: GPU {arch} not in {compiled}, fallback CPU")
				device = torch.device('cpu')
		except Exception:
			pass
	print('[step6] device =', device)

	op, val = parse_query(args.query)
	unet = load_unet(args.unet, device)

	# collect dicom files under root
	dcm_files = []
	for r,_,files in os.walk(args.dicom_root):
		for n in files:
			if n.lower().endswith('.dcm'):
				dcm_files.append(os.path.join(r,n))
	if not dcm_files:
		print('[step6] no DICOM files under', args.dicom_root); return

	rows = []
	for p in dcm_files:
		try:
			ok, dmm, mask, spacing = process_series(p, unet, device, op, val)
			rows.append({'dicom_path': p, 'pred_diameter_mm': dmm, 'ok': ok, 'mask': mask, 'spacing': spacing})
		except Exception as e:
			print('[step6] fail:', p, e)

	ensure_dir(args.out)
	# rank by diameter mm (descending), filter by query condition
	df = pd.DataFrame([{k:v for k,v in r.items() if k not in ('mask','spacing')} for r in rows])
	df_hit = df[df['ok']].sort_values('pred_diameter_mm', ascending=False).head(args.topk).reset_index(drop=True)
	df_hit.to_csv(os.path.join(args.out, 'topk_hits.csv'), index=False)
	print('[step6] hits:', len(df_hit))

	# export DICOM and NIfTI masks
	for i, (_, rec) in enumerate(df_hit.iterrows()):
		p = rec['dicom_path']
		shutil.copy2(p, os.path.join(args.out, f'hit_{i:03d}.dcm'))
		mask = next(r['mask'] for r in rows if r['dicom_path']==p)
		spacing = next(r['spacing'] for r in rows if r['dicom_path']==p)
		out_nii = os.path.join(args.out, f'hit_{i:03d}_mask.nii.gz')
		save_nii_from_mask(mask*255, spacing, out_nii)
		print('[step6] wrote:', out_nii)

	# optionally launch 3D Slicer
	if args.slicer and os.path.exists(args.slicer):
		# Open folder; Slicer will list files for loading
		try:
			print('[step6] launching 3D Slicer...')
			subprocess.Popen([args.slicer, args.out])
		except Exception as e:
			print('[step6] slicer launch failed:', e)

	print('[step6] done ->', args.out)


if __name__ == '__main__':
	main()



