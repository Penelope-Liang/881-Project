import argparse, os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from step5.dataset import ContrastiveRoiDataset
from step5.models import ImgEncoder


def collate_fn(batch):
	imgs, ids, lens = zip(*batch)  # ids/lens unused here
	return torch.stack(imgs,0)


def main():
	ap = argparse.ArgumentParser(description='Step5: build image embeddings for retrieval')
	ap.add_argument('--roi-csv', required=True)
	ap.add_argument('--patients-file', required=True)
	ap.add_argument('--model', required=True)
	ap.add_argument('--out-dir', required=True)
	ap.add_argument('--img-size', type=int, default=256)
	ap.add_argument('--emb', type=int, default=128)
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

	ds = ContrastiveRoiDataset(args.roi_csv, patients_file=args.patients_file, image_size=(args.img_size,args.img_size))
	loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=lambda b: (torch.stack([x[0] for x in b],0),))

	ckpt = torch.load(args.model, map_location='cpu')
	img_enc = ImgEncoder(out_dim=args.emb)
	img_enc.load_state_dict(ckpt['img'])
	img_enc = img_enc.to(device).eval()

	all_emb = []
	with torch.no_grad():
		for (imgs,) in tqdm(loader, desc='Embedding', dynamic_ncols=True):
			imgs = imgs.to(device).float()
			z = img_enc(imgs).detach().cpu().numpy()
			all_emb.append(z)

	emb = np.concatenate(all_emb, axis=0)
	os.makedirs(args.out_dir, exist_ok=True)
	np.save(os.path.join(args.out_dir, 'img_embeds.npy'), emb)
	# meta table
	meta = ds.df[['dicom_path','diameter_mm']].copy()
	meta.to_parquet(os.path.join(args.out_dir, 'meta.parquet'), index=False)
	print('[step5] wrote', args.out_dir)


if __name__ == '__main__':
	main()
