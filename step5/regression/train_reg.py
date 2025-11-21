import argparse, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from step5.regression.dataset_reg import RoiRegressionDataset
from step5.regression.models_reg import ResNetRegBin


def collection(batch):
	"""collection function collates batch data for DataLoader in training

	args:
		batch: list of tuples from dataset __getitem__

	returns:
		tuple of stacked_images, diameter_labels, class_labels, paths_list
	"""
	imgs, diameters, clss, paths, bboxes = zip(*batch)
	return torch.stack(imgs,0), torch.tensor(diameters), torch.tensor(clss), list(paths)


def main():
	"""main function loads ROI dataset, trains ResNetRegBin model, 
	performs training and validation, saves best model checkpoint.
	"""
	parser = argparse.ArgumentParser(description="Step5A: diameter regression + bin classification")
	parser.add_argument("--roi-csv", required=True)
	parser.add_argument("--train-patients", required=True)
	parser.add_argument("--val-patients", required=True)
	parser.add_argument("--epochs", type=int, default=15)
	parser.add_argument("--bs", type=int, default=256)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--img-size", type=int, default=256)
	parser.add_argument("--out", dest="out_dir", default="outputs/step5_reg")
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
	print("[step5A] Device =", device)

	train_dataset = RoiRegressionDataset(args.roi_csv, patients_file=args.train_patients, img_size=(args.img_size,args.img_size))
	validation_dataset   = RoiRegressionDataset(args.roi_csv, patients_file=args.val_patients,   img_size=(args.img_size,args.img_size))
	train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0, collate_fn=collection)
	validation_loader   = DataLoader(validation_dataset,   batch_size=args.bs, shuffle=False, num_workers=0, collate_fn=collection)
	print("[step5A] Dataset sizes:", len(train_dataset), len(validation_dataset))

	model = ResNetRegBin(pretrained=True).to(device)
	mae = nn.L1Loss()
	ce  = nn.CrossEntropyLoss()
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

	best_score = 1e9
	os.makedirs(args.out_dir, exist_ok=True)
	checkpoint_path = os.path.join(args.out_dir, "reg_best.path")

	for ep in range(1, args.epochs+1):
		model.train(); loss_sum=0.0
		for imgs, diameters, clss, _ in tqdm(train_loader, desc=f"Train ep {ep}", dynamic_ncols=True):
			imgs=imgs.to(device).float(); diameters=diameters.to(device).float(); clss=clss.to(device).long()
			opt.zero_grad()
			pred_d, pred_c = model(imgs)
			loss = mae(pred_d, diameters) + 0.2*ce(pred_c, clss)
			loss.backward(); opt.step()
			loss_sum += loss.item()*imgs.size(0)
		train_loss= loss_sum/max(1,len(train_dataset))

		# validate
		model.eval(); mae_sum=0.0; acc_cnt=0; n=0
		with torch.no_grad():
			for imgs, diameters, clss, _ in tqdm(validation_loader, desc=f"Valid ep {ep}", dynamic_ncols=True):
				imgs=imgs.to(device).float(); diameters=diameters.to(device).float(); clss=clss.to(device).long()
				pred_d, pred_c = model(imgs)
				mae_sum += nn.functional.l1_loss(pred_d, diameters, reduction="sum").item()
				acc_cnt += (pred_c.argmax(1)==clss).sum().item()
				n += imgs.size(0)
		validation_mae = mae_sum / max(1,n)
		validation_accuracy = acc_cnt / max(1,n)
		print(f"[step5A] ep {ep}: train_loss={train_loss:.4f} validation_mae={validation_mae:.3f}mm validation_accuracy={validation_accuracy:.3f}")
		score = validation_mae - 2.0 * validation_accuracy
		if score < best_score:
			best_score = score
			torch.save({"model": model.state_dict(), "cfg": vars(args)}, checkpoint_path)
			print("[step5A] saved best to", checkpoint_path)

	print("[step5A] done")


if __name__ == "__main__":
	main()
