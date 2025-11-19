import argparse
import os
import ast
import torch
from PIL import Image
import numpy as np
from pydicom import dcmread
from skimage.transform import resize as sk_resize

from step4.models.unet import UNet


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


def main():
	"""main function runs UNet inference on ROI crops, loads ROI CSV, loads UNet model checkpoint, 
	processes ROI crops from dcm files, runs inference to generate segmentation predictions, 
	saves input images and prediction masks to output directory.
	"""
	parser = argparse.ArgumentParser(description="Step4: UNet inference (ROI crops)")
	parser.add_argument("--roi-csv", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--out", dest="out_dir", required=True)
	parser.add_argument("--img-size", type=int, default=256)
	parser.add_argument("--limit", type=int, default=50)
	args = parser.parse_args()

	import pandas as pd
	df = pd.read_csv(args.roi_csv).head(args.limit)
	os.makedirs(args.out_dir, exist_ok=True)

	checkpoint = torch.load(args.model, map_location="cpu")
	model = UNet(in_channels=1, base=32)
	model.load_state_dict(checkpoint["model"])
	model.eval()

	pad = 16
	for i, row in df.reset_index(drop=True).iterrows():
		dcm_dataset = dcmread(row["dicom_path"])
		img = normalize_img(dcm_dataset.pixel_array.astype(np.float32))
		points = ast.literal_eval(row["points_json"]) if isinstance(row["points_json"], str) else row["points_json"]
		xs = [p[0] for p in points]
		ys = [p[1] for p in points]
		xmin, ymin, xmax, ymax = min(xs)-pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
		xmin = max(0, xmin)
		ymin = max(0, ymin)
		xmax = min(img.shape[1]-1, xmax)
		ymax = min(img.shape[0]-1, ymax)
		crop = img[ymin:ymax+1, xmin:xmax+1]
		crop_resized = sk_resize(crop, (args.img_size, args.img_size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
		input_tensor = torch.from_numpy(crop_resized[None, None, ...])
		with torch.no_grad():
			logits = model(input_tensor)
			prob = torch.sigmoid(logits)[0, 0].numpy()
		Image.fromarray((crop_resized * 255).astype(np.uint8)).save(os.path.join(args.out_dir, f"{i:03d}_img.png"))
		Image.fromarray((prob * 255).astype(np.uint8)).save(os.path.join(args.out_dir, f"{i:03d}_pred.png"))
	print("[step4] inference done")


if __name__ == "__main__":
	main()
