import argparse, os, re, json, shutil
import numpy as np
import pandas as pd

import torch
from pydicom import dcmread
import SimpleITK as sitk

from step4.models.unet import UNet


def parse_query(q: str):
	s = q.strip().lower().replace(' ', '')
	m = re.match(r'(>=|<=|>|<|=)?([0-9]+\.?[0-9]*)mm', s)
	if not m:
		raise ValueError(f"Invalid query: {q}")
	op = m.group(1) or '>='
	val = float(m.group(2))
	return op, val


def match_rows(df: pd.DataFrame, op: str, val: float):
	c = df['pred_diameter_mm']
	if op == '>=': return df[c>=val]
	if op == '>':  return df[c>val]
	if op == '<=': return df[c<=val]
	if op == '<':  return df[c<val]
	if op == '=':  return df[np.isclose(c, val, atol=0.1)]
	raise ValueError(op)


def save_nii_from_mask(mask2d: np.ndarray, spacing_xy: tuple, out_path: str):
	img = sitk.GetImageFromArray(mask2d.astype(np.uint8))
	# set spacing (x,y) -> (col,row) to SITK (x,y)
	img.SetSpacing((float(spacing_xy[1]), float(spacing_xy[0])))
	sitk.WriteImage(img, out_path)


def load_unet(ckpt_p: str, device: torch.device):
	ckpt = torch.load(ckpt_p, map_location='cpu')
	model = UNet(in_ch=1, base=32).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	return model


def norm_img(arr):
	vmin, vmax = np.percentile(arr, [5,95])
	return np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)


def main():
	ap = argparse.ArgumentParser(description='Step6: query -> TopK slices (Step5A) -> UNet mask -> export for 3D Slicer')
	ap.add_argument('--pred-csv', required=True, help='outputs/step5_reg/test_pred/pred_regression.csv')
	ap.add_argument('--query', required=True, help='e.g., ">=6mm"')
	ap.add_argument('--topk', type=int, default=10)
	ap.add_argument('--unet', required=True, help='outputs/step4/unet_best.pth')
	ap.add_argument('--out', required=True)
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

	df = pd.read_csv(args.pred_csv)
	op, val = parse_query(args.query)
	df_hit = match_rows(df, op, val).copy()
	if df_hit.empty:
		print('[step6] no hits for', args.query)
		return
	df_hit = df_hit.sort_values('pred_diameter_mm', ascending=False).head(args.topk).reset_index(drop=True)

	os.makedirs(args.out, exist_ok=True)
	df_hit.to_csv(os.path.join(args.out, 'topk_hits.csv'), index=False)

	unet = load_unet(args.unet, device)

	for i, row in df_hit.iterrows():
		dcm_path = row['dicom_path']
		if not os.path.exists(dcm_path):
			print('[step6] missing:', dcm_path); continue
		# copy dicom
		shutil.copy2(dcm_path, os.path.join(args.out, f'hit_{i:03d}.dcm'))
		# run UNet on full slice
		ds = dcmread(dcm_path)
		px = ds.pixel_array.astype(np.float32)
		nm = norm_img(px)
		import cv2
		img_r = cv2.resize(nm, (256,256), interpolation=cv2.INTER_LINEAR)
		inp = torch.from_numpy(img_r[None,None,...]).to(device)
		with torch.no_grad():
			prob = torch.sigmoid(unet(inp))[0,0].detach().cpu().numpy()
		# upsample back
		mask = (cv2.resize(prob, (px.shape[1], px.shape[0]), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
		# write NIfTI/NRRD for Slicer (we choose NIfTI .nii.gz)
		# spacing from DICOM
		spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])) if hasattr(ds, 'PixelSpacing') else (1.0,1.0)
		out_nii = os.path.join(args.out, f'hit_{i:03d}_mask.nii.gz')
		save_nii_from_mask(mask*255, spacing, out_nii)
		print('[step6] wrote:', out_nii)

	print('[step6] wrote hits CSV and DICOM+NIfTI to', args.out)


if __name__ == '__main__':
	main()



