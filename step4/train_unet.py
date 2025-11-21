import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from step4.models.unet import UNet
from step4.dataset_roi import Roidataset
from PIL import Image
import numpy as np
from tqdm.auto import tqdm


try:
	def progress(it, **kw):
		"""progress function wraps iterator with tqdm progress bar if available

		args:
			it: iterator to wrap
			**kw: keyword arguments passed to tqdm

		returns:
			iterator with progress bar if tqdm available, otherwise original iterator
		"""
		return tqdm(it, **kw)
except Exception:
	def progress(it, **kw):
		"""progress function wraps iterator with tqdm progress bar if available

		args:
			it: iterator to wrap
			**kw: keyword arguments passed to tqdm

		returns:
			iterator with progress bar if tqdm available, otherwise original iterator
		"""
		return it


def dice_loss(logits, target, eps: float = 1e-6):
	"""dice_loss function computes dice loss for binary segmentation

	args:
		logits: predicted logits tensor of shape
		target: target binary mask tensor of shape
		eps: small epsilon value to avoid division by zero.

	returns:
		dice loss value (1 - dice coefficient)
	"""
	pred = torch.sigmoid(logits)
	inter = (pred * target).sum(dim=(2, 3))
	sum_ = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
	dice = (2 * inter + eps) / (sum_ + eps)
	return 1 - dice.mean()


def save_preview(img_tensor, mask_tensor, pred_tensor, out_dir, step):
	"""save_preview function saves preview images of input, ground truth mask, and prediction

	args:
		img_tensor: input img tensor of shape
		mask_tensor: ground truth mask tensor of shape
		pred_tensor: predicted logits tensor of shape
		out_dir: output directory path for saving preview images
		step: step number for filename
	"""
	os.makedirs(out_dir, exist_ok=True)
	img = (img_tensor[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
	mask = (mask_tensor[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
	prob = (torch.sigmoid(pred_tensor[0, 0]).detach().cpu().numpy() * 255).astype(np.uint8)
	Image.fromarray(img).save(os.path.join(out_dir, f"step_{step:04d}_img.png"))
	Image.fromarray(mask).save(os.path.join(out_dir, f"step_{step:04d}_gt.png"))
	Image.fromarray(prob).save(os.path.join(out_dir, f"step_{step:04d}_pred.png"))


def main():
	"""main function trains loads ROI dataset, split train, validation sets, trains UNet model with dice loss and BCE loss,
	saves best model checkpoint and preview images during training.
	"""
	parser = argparse.ArgumentParser(description="Step4: Train UNet on ROI crops")
	parser.add_argument("--roi-csv", required=True, help="outputs/step2/roi_with_diam.csv")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--bs", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--img-size", type=int, default=256)
	parser.add_argument("--limit", type=int, default=5000)
	parser.add_argument("--validation-split", type=float, default=0.1)
	parser.add_argument("--out", dest="out_dir", default="outputs/step4")
	parser.add_argument("--train-patients", type=str, default=None, help="Path to train_patients.txt")
	parser.add_argument("--val-patients", type=str, default=None, help="Path to validation_patients.txt")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		try:
			cap = torch.cuda.get_device_capability(0)
			arch = f"sm_{cap[0]}{cap[1]}"
			compiled = torch.cuda.get_arch_list()
			if arch not in compiled:
				print(f"[step4] WARNING: GPU {arch} not in compiled arches {compiled}, fallback CPU")
				device = torch.device("cpu")
		except Exception:
			pass
	print(f"[step4] Device={device}")

	if args.train_patients or args.val_patients:
		train_dataset = Roidataset(args.roi_csv, img_size=(args.img_size, args.img_size), limit=None if args.limit==0 else args.limit, patients_file=args.train_patients)
		validation_dataset= Roidataset(args.roi_csv, img_size=(args.img_size, args.img_size), limit=None if args.limit==0 else min(args.limit, 10000), patients_file=args.val_patients)
		print(f"[step4] dataset (patients): train={len(train_dataset)} validation={len(validation_dataset)}")
	else:
		dataset = Roidataset(args.roi_csv, img_size=(args.img_size, args.img_size), limit=None if args.limit==0 else args.limit)
		validation_len = max(1, int(len(dataset) * args.validation_split))
		train_len = len(dataset) - validation_len
		train_dataset, validation_dataset= random_split(dataset, [train_len, validation_len])
		print(f"[step4] dataset: train = {len(train_dataset)} validation = {len(validation_dataset)} total={len(dataset)}")

	train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0)
	validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=False, num_workers=0)

	model = UNet(in_channels=1, base=32).to(device)
	bce = nn.BCEWithLogitsLoss()
	opt = optim.AdamW(model.parameters(), lr=args.lr)
	scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

	best_validation_loss = 1e9
	os.makedirs(args.out_dir, exist_ok=True)
	checkpoint_path = os.path.join(args.out_dir, "unet_best.pth")
	preview_dir = os.path.join(args.out_dir, "previews")

	step = 0
	for ep in range(1, args.epochs + 1):
		model.train()
		train_sum = 0.0
		for img, msk in progress(train_loader, total=len(train_loader), desc=f"Train ep {ep}", leave=False, dynamic_ncols=True):
			img = img.to(device)
			msk = msk.to(device)
			opt.zero_grad(set_to_none=True)
			with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
				logits = model(img)
				loss = bce(logits, msk) + dice_loss(logits, msk)
			scaler.scale(loss).backward()
			scaler.step(opt)
			scaler.update()
			train_sum += loss.item() * img.size(0)
			if step % 200 == 0:
				save_preview(img, msk, logits, preview_dir, step)
			step += 1
		train_loss = train_sum / max(1, len(train_loader.dataset))

		model.eval()
		validation_sum = 0.0
		with torch.no_grad():
			for img, msk in progress(validation_loader, total=len(validation_loader), desc=f"Valid ep {ep}", leave=False, dynamic_ncols=True):
				img = img.to(device)
				msk = msk.to(device)
				logits = model(img)
				loss = bce(logits, msk) + dice_loss(logits, msk)
				validation_sum += loss.item() * img.size(0)
		validation_loss = validation_sum / max(1, len(validation_loader.dataset))
		print(f"[step4] epoch {ep}: train={train_loss:.4f} validation={validation_loss:.4f}")
		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			torch.save({"model": model.state_dict(), "cfg": vars(args)}, checkpoint_path)
			print(f"[step4] saved best: {checkpoint_path}")

	print("[step4] done")


if __name__ == "__main__":
	main()
