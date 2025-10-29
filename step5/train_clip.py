import argparse, os, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from step5.dataset import ContrastiveRoiDataset
from step5.models import ImgEncoder, TxtEncoder, clip_loss


def recall_at_k(img_z, txt_z, k: int = 10):
	# cosine sim
	img_z = torch.nn.functional.normalize(img_z, dim=-1)
	txt_z = torch.nn.functional.normalize(txt_z, dim=-1)
	sim = img_z @ txt_z.t()
	topk = sim.topk(k, dim=1).indices
	target = torch.arange(img_z.size(0), device=img_z.device).unsqueeze(1)
	return (topk == target).any(dim=1).float().mean().item()


def collate_fn(batch):
	imgs, ids, lens = zip(*batch)
	return torch.stack(imgs,0), torch.stack(ids,0), torch.stack(lens,0)


def main():
	ap = argparse.ArgumentParser(description='Step5: CLIP-style contrastive training')
	ap.add_argument('--roi-csv', required=True)
	ap.add_argument('--train-patients', required=True)
	ap.add_argument('--val-patients', required=True)
	ap.add_argument('--epochs', type=int, default=10)
	ap.add_argument('--bs', type=int, default=256)
	ap.add_argument('--lr', type=float, default=1e-3)
	ap.add_argument('--emb', type=int, default=128)
	ap.add_argument('--out', default='outputs/step5')
	ap.add_argument('--img-size', type=int, default=256)
	ap.add_argument('--pretrained', action='store_true')
	args = ap.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if device.type == 'cuda':
		try:
			cap = torch.cuda.get_device_capability(0)
			arch = f"sm_{cap[0]}{cap[1]}"
			compiled = torch.cuda.get_arch_list()
			if arch not in compiled:
				print(f"[step5] WARN: GPU {arch} not in compiled arches {compiled}, fallback CPU")
				device = torch.device('cpu')
		except Exception:
			pass
	print('[step5] device =', device)

	train_ds = ContrastiveRoiDataset(args.roi_csv, patients_file=args.train_patients, image_size=(args.img_size,args.img_size))
	val_ds   = ContrastiveRoiDataset(args.roi_csv, patients_file=args.val_patients,   image_size=(args.img_size,args.img_size), vocab=train_ds.vocab)
	print('[step5] dataset sizes:', len(train_ds), len(val_ds))

	train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0, collate_fn=collate_fn)
	val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=0, collate_fn=collate_fn)

	img_enc = ImgEncoder(out_dim=args.emb, pretrained=args.pretrained).to(device)
	txt_enc = TxtEncoder(vocab=train_ds.vocab, out_dim=args.emb).to(device)
	opt = torch.optim.AdamW(list(img_enc.parameters())+list(txt_enc.parameters()), lr=args.lr)

	best_val = -1.0
	os.makedirs(args.out, exist_ok=True)
	ckpt_p = os.path.join(args.out, 'clip_best.pth')

	for ep in range(1, args.epochs+1):
		img_enc.train(); txt_enc.train()
		loss_sum = 0.0
		for imgs, ids, lens in tqdm(train_loader, desc=f'Train ep {ep}', dynamic_ncols=True):
			imgs = imgs.to(device).float()
			ids  = ids.to(device).long()
			lens = lens.to(device).long()
			opt.zero_grad()
			iz = img_enc(imgs)
			tz = txt_enc(ids, lens)
			loss = clip_loss(iz, tz, temperature=0.05)
			loss.backward()
			opt.step()
			loss_sum += loss.item() * imgs.size(0)
		train_loss = loss_sum / max(1, len(train_ds))

		# val recall@10
		img_enc.eval(); txt_enc.eval()
		with torch.no_grad():
			all_iz, all_tz = [], []
			for imgs, ids, lens in tqdm(val_loader, desc=f'Valid ep {ep}', dynamic_ncols=True):
				imgs = imgs.to(device).float(); ids=ids.to(device).long(); lens=lens.to(device).long()
				all_iz.append(img_enc(imgs))
				all_tz.append(txt_enc(ids, lens))
			iz = torch.cat(all_iz,0); tz = torch.cat(all_tz,0)
			rec10 = recall_at_k(iz, tz, k=10)
		print(f"[step5] ep {ep}: train_loss={train_loss:.4f} val_recall@10={rec10:.4f}")
		if rec10 > best_val:
			best_val = rec10
			torch.save({'img': img_enc.state_dict(), 'txt': txt_enc.state_dict(), 'vocab': train_ds.vocab, 'cfg': vars(args)}, ckpt_p)
			print('[step5] saved best to', ckpt_p)

	print('[step5] done')


if __name__ == '__main__':
	main()
