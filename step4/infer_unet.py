import argparse
import os
import ast

import torch
from PIL import Image
import numpy as np
from pydicom import dcmread
from skimage.draw import polygon as sk_polygon
from skimage.transform import resize as sk_resize

from step4.models.unet import UNet


def norm_img(arr):
	vmin, vmax = np.percentile(arr, [5, 95])
	arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
	return arr


def main():
	parser = argparse.ArgumentParser(description='Step4: UNet inference (ROI crops)')
	parser.add_argument('--roi-csv', required=True)
	parser.add_argument('--model', required=True)
	parser.add_argument('--out', required=True)
	parser.add_argument('--img-size', type=int, default=256)
	parser.add_argument('--limit', type=int, default=50)
	args = parser.parse_args()

	import pandas as pd
	df = pd.read_csv(args.roi_csv).head(args.limit)
	os.makedirs(args.out, exist_ok=True)

	ckpt = torch.load(args.model, map_location='cpu')
	model = UNet(in_ch=1, base=32)
	model.load_state_dict(ckpt['model'])
	model.eval()

	for i, row in df.reset_index(drop=True).iterrows():
		ds = dcmread(row['dicom_path'])
		img = norm_img(ds.pixel_array.astype(np.float32))
		pts = ast.literal_eval(row['points_json']) if isinstance(row['points_json'], str) else row['points_json']
		xs = [p[0] for p in pts]
		ys = [p[1] for p in pts]
		xmin, ymin, xmax, ymax = min(xs)-16, min(ys)-16, max(xs)+16, max(ys)+16
		xmin = max(0, xmin)
		ymin = max(0, ymin)
		xmax = min(img.shape[1]-1, xmax)
		ymax = min(img.shape[0]-1, ymax)
		crop = img[ymin:ymax+1, xmin:xmax+1]
		crop_r = sk_resize(crop, (args.img_size, args.img_size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
		inp = torch.from_numpy(crop_r[None, None, ...])
		with torch.no_grad():
			logits = model(inp)
			prob = torch.sigmoid(logits)[0, 0].numpy()
		Image.fromarray((crop_r * 255).astype(np.uint8)).save(os.path.join(args.out, f"{i:03d}_img.png"))
		Image.fromarray((prob * 255).astype(np.uint8)).save(os.path.join(args.out, f"{i:03d}_pred.png"))
	print('[step4] inference done')


if __name__ == '__main__':
	main()






