import argparse, os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from step5.regression.dataset_reg import RoiRegressionDataset, bin_label
from step5.regression.models_reg import ResNetRegBin


def collate_fn(batch):
	imgs, diams, clss, paths, bboxes = zip(*batch)
	return torch.stack(imgs,0), list(paths), list(bboxes)


def main():
	ap = argparse.ArgumentParser(description='Step5A: run regression model on a split')
	ap.add_argument('--roi-csv', required=True)
	ap.add_argument('--patients-file', required=True)
	ap.add_argument('--model', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--img-size', type=int, default=256)
	args = ap.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if device.type == 'cuda':
		try:
			cap = torch.cuda.get_device_capability(0)
			arch = f"sm_{cap[0]}{cap[1]}"
			compiled = torch.cuda.get_arch_list()
			if arch not in compiled:
				print(f"[step5A] WARN: GPU {arch} not in compiled arches {compiled}, fallback CPU")
				device = torch.device('cpu')
		except Exception:
			pass
	print('[step5A] device =', device)

	ds = RoiRegressionDataset(args.roi_csv, patients_file=args.patients_file, image_size=(args.img_size,args.img_size))
	loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_fn)

	ckpt = torch.load(args.model, map_location='cpu')
	model = ResNetRegBin(pretrained=False)
	model.load_state_dict(ckpt['model'])
	model = model.to(device).eval()

	pred_rows = []
	with torch.no_grad():
		for imgs, paths, bboxes in tqdm(loader, desc='Predict', dynamic_ncols=True):
			imgs = imgs.to(device).float()
			pd_d, pd_c = model(imgs)
			for i, p in enumerate(paths):
				pred_rows.append({
					'dicom_path': p,
					'pred_diameter_mm': float(pd_d[i].cpu().item()),
					'pred_bin': int(pd_c[i].argmax().cpu().item()),
				})

	os.makedirs(args.out, exist_ok=True)
	out_csv = os.path.join(args.out, 'pred_regression.csv')
	pd.DataFrame(pred_rows).to_csv(out_csv, index=False)
	print('[step5A] wrote', out_csv)


if __name__ == '__main__':
	main()
