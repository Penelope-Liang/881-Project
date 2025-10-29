import argparse, os, json
import numpy as np
import pandas as pd
import torch
from step5.models import TxtEncoder
from step5.dataset import build_vocab, tokenize
from step5.text_templates import all_query_list


def cosine_sim(a, b):
	a = a / (np.linalg.norm(a, axis=1, keepdims=True)+1e-9)
	b = b / (np.linalg.norm(b, axis=1, keepdims=True)+1e-9)
	return a @ b.T


def main():
	ap = argparse.ArgumentParser(description='Step5: semantic query Top-K retrieval')
	ap.add_argument('--emb-dir', required=True, help='Folder with img_embeds.npy and meta.parquet')
	ap.add_argument('--model', required=True, help='CLIP checkpoint (clip_best.pth)')
	ap.add_argument('--query', required=True)
	ap.add_argument('--topk', type=int, default=10)
	ap.add_argument('--out', required=True)
	args = ap.parse_args()

	emb = np.load(os.path.join(args.emb_dir, 'img_embeds.npy'))
	meta = pd.read_parquet(os.path.join(args.emb_dir, 'meta.parquet'))

	ckpt = torch.load(args.model, map_location='cpu')
	vocab = ckpt.get('vocab') or build_vocab(all_query_list())
	txt = TxtEncoder(vocab=vocab, out_dim=emb.shape[1])
	txt.load_state_dict(ckpt['txt'])
	txt.eval()

	ids, length = tokenize(args.query, vocab, max_len=24)
	with torch.no_grad():
		z = txt(ids[None,...], length[None,...])[0].numpy()
	sim = cosine_sim(emb, z[None,...])[:,0]
	idx = np.argsort(-sim)[:args.topk]
	res = meta.iloc[idx].copy()
	res['sim'] = sim[idx]
	os.makedirs(args.out, exist_ok=True)
	res.to_csv(os.path.join(args.out, 'topk.csv'), index=False)
	print(res[['dicom_path','diameter_mm','sim']].head(args.topk))
	print('[step5] wrote', os.path.join(args.out, 'topk.csv'))


if __name__ == '__main__':
	main()
