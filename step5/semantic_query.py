import argparse
import os
import numpy as np
import pandas as pd
import torch
from step5.models import TxtEncoder
from step5.dataset import build_vocab, tokenize
from step5.text_templates import all_query_list


def cosine_sim(a, b):
	"""cosine_sim function computes cosine similarity matrix between two sets of vectors

	args:
		a: first set of vectors
		b: second set of vectors

	returns:
		cosine similarity matrix
	"""
	a = a / (np.linalg.norm(a, axis=1, keepdims=True)+1e-9)
	b = b / (np.linalg.norm(b, axis=1, keepdims=True)+1e-9)
	return a @ b.T


def main():
	"""main function performs semantic query retrieval using CLIP model
	"""
	parser = argparse.ArgumentParser(description='Step5: semantic query Top-K retrieval')
	parser.add_argument('--emb-dir', required=True, help='Folder with img_embeds.npy and meta.parquet')
	parser.add_argument('--model', required=True, help='CLIP checkpoint (clip_best.pth)')
	parser.add_argument('--query', required=True)
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--out', dest='out_dir', required=True)
	args = parser.parse_args()

	emb = np.load(os.path.join(args.emb_dir, 'img_embeds.npy'))
	meta = pd.read_parquet(os.path.join(args.emb_dir, 'meta.parquet'))

	checkpoint = torch.load(args.model, map_location='cpu')
	vocab = checkpoint.get('vocab') or build_vocab(all_query_list())
	txt = TxtEncoder(vocab=vocab, out_dim=emb.shape[1])
	txt.load_state_dict(checkpoint['txt'])
	txt.eval()

	ids, length = tokenize(args.query, vocab, max_len=24)
	with torch.no_grad():
		z = txt(ids[None,...], length[None,...])[0].numpy()
	sim = cosine_sim(emb, z[None,...])[:,0]
	idx = np.argsort(-sim)[:args.topk]
	res = meta.iloc[idx].copy()
	res['sim'] = sim[idx]
	os.makedirs(args.out_dir, exist_ok=True)
	res.to_csv(os.path.join(args.out_dir, 'topk.csv'), index=False)
	print(res[['dicom_path','diameter_mm','sim']].head(args.topk))
	print('[step5] wrote', os.path.join(args.out_dir, 'topk.csv'))


if __name__ == '__main__':
	main()
