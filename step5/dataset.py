import ast
import re
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from skimage.transform import resize as sk_resize

from step5.text_templates import size_text_for_diameter, sample_text_for_diameter


def _patient_from_path(path: str) -> Optional[str]:
	m = re.search(r"(LIDC-IDRI-\d{4})", path.replace('\\','/'))
	return m.group(1) if m else None


def build_vocab(texts: List[str]) -> Dict[str, int]:
	v = {"<pad>":0, "<unk>":1}
	for t in texts:
		for tok in t.lower().split():
			if tok not in v:
				v[tok] = len(v)
	return v


def tokenize(text: str, vocab: Dict[str,int], max_len: int = 24):
	toks = [vocab.get(tok, vocab["<unk>"]) for tok in text.lower().split()][:max_len]
	length = len(toks)
	if length < max_len:
		toks = toks + [vocab["<pad>"]]*(max_len-length)
	return torch.tensor(toks, dtype=torch.long), torch.tensor(length, dtype=torch.long)


class ContrastiveRoiDataset(Dataset):
	def __init__(self, roi_csv: str, patients_file: Optional[str], image_size: Tuple[int,int]=(256,256), max_len: int = 24, vocab: Optional[Dict[str,int]] = None):
		self.df = pd.read_csv(roi_csv)
		if patients_file:
			ps = set([l.strip() for l in open(patients_file,'r',encoding='utf-8') if l.strip()])
			self.df = self.df[self.df['dicom_path'].astype(str).apply(lambda p: _patient_from_path(p) in ps)]
		self.df = self.df.dropna(subset=['dicom_path','points_json','diameter_mm']).reset_index(drop=True)
		self.image_size = image_size
		self.max_len = max_len
		# shared vocab
		if vocab is None:
			texts = [size_text_for_diameter(d) for d in self.df['diameter_mm'].tolist()]
			self.vocab = build_vocab(texts)
		else:
			self.vocab = vocab

	def __len__(self):
		return len(self.df)

	def _load_img(self, p: str) -> np.ndarray:
		ds = dcmread(p)
		arr = ds.pixel_array.astype(np.float32)
		vmin, vmax = np.percentile(arr, [5, 95])
		arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
		return arr

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		img = self._load_img(row['dicom_path'])
		H, W = img.shape
		pts = ast.literal_eval(row['points_json']) if isinstance(row['points_json'], str) else row['points_json']
		xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
		xmin, ymin, xmax, ymax = max(0,min(xs)-16), max(0,min(ys)-16), min(W-1,max(xs)+16), min(H-1,max(ys)+16)
		crop = img[ymin:ymax+1, xmin:xmax+1]
		crop = sk_resize(crop, self.image_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

		text = sample_text_for_diameter(float(row['diameter_mm']), rnd=True)
		ids, length = tokenize(text, self.vocab, self.max_len)
		return torch.from_numpy(crop[None,...]), ids, length
