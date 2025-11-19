import argparse, os, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from step5.dataset import ContrastiveRoiDataset
from step5.models import ImgEncoder, TxtEncoder, clip_loss


def recall_at_k(img_z, txt_z, k: int = 10):
	"""recall_at_k function computes Recall@K metric for img-text retrieval

	args:
		img_z: normalized img embeddings of shape (batch, dim)
		txt_z: normalized text embeddings of shape (batch, dim)
		k: number of top results to consider (default: 10)

	returns:
		recall@k score
	"""
	img_z = torch.nn.functional.normalize(img_z, dim=-1)
	txt_z = torch.nn.functional.normalize(txt_z, dim=-1)
	sim = img_z @ txt_z.t()
	topk = sim.topk(k, dim=1).indices
	target = torch.arange(img_z.size(0), device=img_z.device).unsqueeze(1)
	return (topk == target).any(dim=1).float().mean().item()


def collection(batch):
	"""collection function collates individual samples into batched tensors

	args:
		batch: list of individual samples from ContrastiveRoiDataset

	returns:
		tuple of batched_images, batched_token_ids, batched_lengths
	"""
	imgs, ids, lens = zip(*batch)
	return torch.stack(imgs,0), torch.stack(ids,0), torch.stack(lens,0)


def main():
	"""main function trains CLIP contrastive model for img-text retrieval
	"""
	parser = argparse.ArgumentParser(description="Step5: CLIP-style contrastive training")
	parser.add_argument("--roi-csv", required=True)
	parser.add_argument("--train-patients", required=True)
	parser.add_argument("--val-patients", required=True)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--bs", type=int, default=256)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--emb", type=int, default=128)
	parser.add_argument("--out", dest="out_dir", default="outputs/step5")
	parser.add_argument("--img-size", type=int, default=256)
	parser.add_argument("--pretrained", action="store_true")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		try:
			cap = torch.cuda.get_device_capability(0)
			arch = f"sm_{cap[0]}{cap[1]}"
			compiled = torch.cuda.get_arch_list()
			if arch not in compiled:
				print(f"[step5] WARN: GPU {arch} not in compiled arches {compiled}, fallback CPU")
				device = torch.device("cpu")
		except Exception:
			pass
	print("[step5] device =", device)

	train_dataset = ContrastiveRoiDataset(args.roi_csv, patients_file=args.train_patients, img_size=(args.img_size,args.img_size))
	validation_dataset   = ContrastiveRoiDataset(args.roi_csv, patients_file=args.val_patients,   img_size=(args.img_size,args.img_size), vocab=train_dataset.vocab)
	print("[step5] dataset sizes:", len(train_dataset), len(validation_dataset))

	train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0, collate_fn=collection)
	validation_loader   = DataLoader(validation_dataset,   batch_size=args.bs, shuffle=False, num_workers=0, collate_fn=collection)

	img_encoder = ImgEncoder(out_dim=args.emb, pretrained=args.pretrained).to(device)
	txt_encoder = TxtEncoder(vocab=train_dataset.vocab, out_dim=args.emb).to(device)
	opt = torch.optim.AdamW(list(img_encoder.parameters())+list(txt_encoder.parameters()), lr=args.lr)

	best_validation_score = -1.0
	os.makedirs(args.out_dir, exist_ok=True)
	checkpoint_path = os.path.join(args.out_dir, "clip_best.pth")

	for ep in range(1, args.epochs+1):
		img_encoder.train(); txt_encoder.train()
		loss_sum = 0.0
		for imgs, ids, lens in tqdm(train_loader, desc=f"Train epoch {ep}", dynamic_ncols=True):
			imgs = imgs.to(device).float()
			ids  = ids.to(device).long()
			lens = lens.to(device).long()
			opt.zero_grad()
			iz = img_encoder(imgs)
			tz = txt_encoder(ids, lens)
			loss = clip_loss(iz, tz, temperature=0.05)
			loss.backward()
			opt.step()
			loss_sum += loss.item() * imgs.size(0)
		train_loss = loss_sum / max(1, len(train_dataset))

		img_encoder.eval()
		txt_encoder.eval()
		with torch.no_grad():
			all_iz, all_tz = [], []
			for imgs, ids, lens in tqdm(validation_loader, desc=f"Valid ep {ep}", dynamic_ncols=True):
				imgs = imgs.to(device).float(); ids=ids.to(device).long(); lens=lens.to(device).long()
				all_iz.append(img_encoder(imgs))
				all_tz.append(txt_encoder(ids, lens))
			iz = torch.cat(all_iz,0); tz = torch.cat(all_tz,0)
			rec10 = recall_at_k(iz, tz, k=10)
		print(f"[step5] ep {ep}: train_loss={train_loss:.4f} validation_recall@10={rec10:.4f}")
		if rec10 > best_validation_score:
			best_validation_score = rec10
			torch.save({"img": img_encoder.state_dict(), "txt": txt_encoder.state_dict(), "vocab": train_dataset.vocab, "cfg": vars(args)}, checkpoint_path)
			print("[step5] Saved best to", checkpoint_path)

	print("[step5] Done")


if __name__ == "__main__":
	main()
