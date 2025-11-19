import argparse, os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from step5.dataset import ContrastiveRoiDataset
from step5.models import ImgEncoder


def collection(batch):
	"""collection function collates batch data for DataLoader

	args:
		batch: list of tuples from dataset __getitem__

	returns:
		stacked tensor of images
	"""
	imgs, ids, lens = zip(*batch)  # ids/lens unused here
	return torch.stack(imgs,0)


def main():
	"""main function loads trained img encoder, processes ROI dataset, generates embeddings for all imgs,
	saves embeddings as numpy array and metadata.
	"""
	parser = argparse.ArgumentParser(description="Step5: Build img embeddings for retrieval")
	parser.add_argument("--roi-csv", required=True)
	parser.add_argument("--patients-file", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--out", dest="out_dir", required=True)
	parser.add_argument("--img-size", type=int, default=256)
	parser.add_argument("--emb", type=int, default=128)
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
	print("[step5] Device =", device)

	dataset = ContrastiveRoiDataset(args.roi_csv, patients_file=args.patients_file, img_size=(args.img_size,args.img_size))
	loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=lambda b: (torch.stack([x[0] for x in b],0),))

	checkpoint = torch.load(args.model, map_location="cpu")
	img_encoder = ImgEncoder(out_dim=args.emb)
	img_encoder.load_state_dict(checkpoint["img"])
	img_encoder = img_encoder.to(device).eval()

	all_emb = []
	with torch.no_grad():
		for (imgs,) in tqdm(loader, desc="Embedding", dynamic_ncols=True):
			imgs = imgs.to(device).float()
			z = img_encoder(imgs).detach().cpu().numpy()
			all_emb.append(z)

	emb = np.concatenate(all_emb, axis=0)
	os.makedirs(args.out_dir, exist_ok=True)
	np.save(os.path.join(args.out_dir, "img_embeds.npy"), emb)
	meta = dataset.df[["dicom_path","diameter_mm"]].copy()
	meta.to_parquet(os.path.join(args.out_dir, "meta.parquet"), index=False)
	print("[step5] Wrote", args.out_dir)


if __name__ == "__main__":
	main()
