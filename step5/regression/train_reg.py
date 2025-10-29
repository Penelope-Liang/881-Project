import argparse, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from step5.regression.dataset_reg import RoiRegressionDataset
from step5.regression.models_reg import ResNetRegBin


def collate_fn(batch):
	imgs, diams, clss, paths = zip(*batch)
	return torch.stack(imgs,0), torch.tensor(diams), torch.tensor(clss), list(paths)


def main():
	ap = argparse.ArgumentParser(description='Step5A: diameter regression + bin classification')
	ap.add_argument('--roi-csv', required=True)
	ap.add_argument('--train-patients', required=True)
	ap.add_argument('--val-patients', required=True)
	ap.add_argument('--epochs', type=int, default=15)
	ap.add_argument('--bs', type=int, default=256)
	ap.add_argument('--lr', type=float, default=1e-3)
	ap.add_argument('--img-size', type=int, default=256)
	ap.add_argument('--out', default='outputs/step5_reg')
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

	train_ds = RoiRegressionDataset(args.roi_csv, patients_file=args.train_patients, image_size=(args.img_size,args.img_size))
	val_ds   = RoiRegressionDataset(args.roi_csv, patients_file=args.val_patients,   image_size=(args.img_size,args.img_size))
	train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0, collate_fn=collate_fn)
	val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=0, collate_fn=collate_fn)
	print('[step5A] dataset sizes:', len(train_ds), len(val_ds))

	model = ResNetRegBin(pretrained=True).to(device)
	mae = nn.L1Loss()
	ce  = nn.CrossEntropyLoss()
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

	best_score = 1e9
	os.makedirs(args.out, exist_ok=True)
	ckpt_p = os.path.join(args.out, 'reg_best.pth')

	for ep in range(1, args.epochs+1):
		model.train(); loss_sum=0.0
		for imgs, diams, clss, _ in tqdm(train_loader, desc=f'Train ep {ep}', dynamic_ncols=True):
			imgs=imgs.to(device).float(); diams=diams.to(device).float(); clss=clss.to(device).long()
			opt.zero_grad()
			pred_d, pred_c = model(imgs)
			loss = mae(pred_d, diams) + 0.2*ce(pred_c, clss)
			loss.backward(); opt.step()
			loss_sum += loss.item()*imgs.size(0)
		train_loss= loss_sum/max(1,len(train_ds))

		# validate
		model.eval(); mae_sum=0.0; acc_cnt=0; n=0
		with torch.no_grad():
			for imgs, diams, clss, _ in tqdm(val_loader, desc=f'Valid ep {ep}', dynamic_ncols=True):
				imgs=imgs.to(device).float(); diams=diams.to(device).float(); clss=clss.to(device).long()
				pred_d, pred_c = model(imgs)
				mae_sum += nn.functional.l1_loss(pred_d, diams, reduction='sum').item()
				acc_cnt += (pred_c.argmax(1)==clss).sum().item()
				n += imgs.size(0)
		val_mae = mae_sum/max(1,n)
		val_acc = acc_cnt/max(1,n)
		print(f"[step5A] ep {ep}: train_loss={train_loss:.4f} val_mae={val_mae:.3f}mm val_acc={val_acc:.3f}")
		# combined score: lower is better
		score = val_mae - 2.0*val_acc
		if score < best_score:
			best_score = score
			torch.save({'model': model.state_dict(), 'cfg': vars(args)}, ckpt_p)
			print('[step5A] saved best to', ckpt_p)

	print('[step5A] done')


if __name__ == '__main__':
	main()



