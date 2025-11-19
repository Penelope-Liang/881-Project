import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from step5.regression.dataset_reg import RoiRegressionDataset
from step5.regression.models_reg import ResNetRegBin


def collection(batch):
	"""collection function collates batch data for DataLoader in prediction

	args:
		batch: list of tuples from dataset __getitem__

	returns:
		tuple of stacked_images, paths_list, bboxes_list
	"""
	imgs, diameters, clss, paths, bboxes = zip(*batch)
	return torch.stack(imgs,0), list(paths), list(bboxes)


def main():
	"""main function loads ROI dataset, trained ResNetRegBin model, 
	performs inference to predict diameter values and classification bins, 
	saves predictions to CSV file.
	"""
	parser = argparse.ArgumentParser(description="Step5A: run regression model on a split")
	parser.add_argument("--roi-csv", required=True)
	parser.add_argument("--patients-file", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--out", dest="out_dir", required=True)
	parser.add_argument("--img-size", type=int, default=256)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		try:
			cap = torch.cuda.get_device_capability(0)
			arch = f"sm_{cap[0]}{cap[1]}"
			compiled = torch.cuda.get_arch_list()
			if arch not in compiled:
				print(f"[step5A] WARNING: GPU {arch} not in compiled arches {compiled}, fallback CPU")
				device = torch.device("cpu")
		except Exception:
			pass
	print("[step5A] device =", device)

	roi_dataset = RoiRegressionDataset(args.roi_csv, patients_file=args.patients_file, img_size=(args.img_size,args.img_size))
	loader = DataLoader(roi_dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collection)

	checkpoint = torch.load(args.model, map_location="cpu")
	model = ResNetRegBin(pretrained=False)
	model.load_state_dict(checkpoint["model"])
	model = model.to(device).eval()

	pred_rows = []
	with torch.no_grad():
		for imgs, paths, bboxes in tqdm(loader, desc="Predict", dynamic_ncols=True):
			imgs = imgs.to(device).float()
			pd_d, pd_c = model(imgs)
			for i, p in enumerate(paths):
				pred_rows.append({
					"dicom_path": p,
					"pred_diameter_mm": float(pd_d[i].cpu().item()),
					"pred_bin": int(pd_c[i].argmax().cpu().item()),
				})

	os.makedirs(args.out_dir, exist_ok=True)
	out_csv = os.path.join(args.out_dir, "pred_regression.csv")
	pd.DataFrame(pred_rows).to_csv(out_csv, index=False)
	print("[step5A] Wrote", out_csv)


if __name__ == "__main__":
	main()
