import ast
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from skimage.transform import resize as sk_resize
from step5.text_templates import size_text_for_diameter, sample_text_for_diameter
from typing import Tuple, Optional, List, Dict


def patient_from_path(path: str) -> Optional[str]:
	"""patient_from_path function extracts patient id from file path

	args:
		path: file path containing patient id

	returns:
		patient id str like "LIDC-IDRI-0001" or None
	"""
	match = re.search(r"(LIDC-IDRI-\d{4})", path.replace("\\","/"))
	return match.group(1) if match else None


def build_vocab(texts: List[str]) -> Dict[str, int]:
	"""build_vocab function builds vocab dict from text corpus

	args:
		texts: list of text to build vocab

	returns:
		vocab dict mapping tokens to integer ids
	"""
	vocab = {"<pad>":0, "<unk>":1}
	for t in texts:
		for tok in t.lower().split():
			if tok not in vocab:
				vocab[tok] = len(vocab)
	return vocab


def tokenize(text: str, vocab: Dict[str,int], max_len: int = 24):
	"""tokenize function converts text to token ids with padding

	args:
		text: input text string to tokenize
		vocab: vocab dict mapping tokens to ids
		max_len: max sequence length

	returns:
		tuple of token_ids tensor, actual_length tensor
	"""
	toks = [vocab.get(tok, vocab["<unk>"]) for tok in text.lower().split()][:max_len]
	toks_len = len(toks)
	if toks_len < max_len:
		toks = toks + [vocab["<pad>"]]*(max_len-toks_len)
	return torch.tensor(toks, dtype=torch.long), torch.tensor(toks_len, dtype=torch.long)


class ContrastiveRoiDataset(Dataset):
	def __init__(self, roi_csv: str, patients_file: Optional[str], img_size: Tuple[int,int]=(256,256), max_len: int = 24, vocab: Optional[Dict[str,int]] = None):
		"""ContrastiveRoiDataset constructor initializes dataset for contrastive learning

		args:
			roi_csv: path to CSV file containing ROI data
			patients_file: path to file containing patient ids
			img_size: target size for resized imgs
			max_len: max text sequence length
			vocab: pre-built vocab dict
		"""
		self.df = pd.read_csv(roi_csv)
		if patients_file:
			patient_set = set([line.strip() for line in open(patients_file,"r",encoding="utf-8") if line.strip()])
			self.df = self.df[self.df["dicom_path"].astype(str).apply(lambda p: patient_from_path(p) in patient_set)]
		self.df = self.df.dropna(subset=["dicom_path","points_json","diameter_mm"]).reset_index(drop=True)
		self.img_size = img_size
		self.max_len = max_len
		if vocab is None:
			texts = [size_text_for_diameter(d) for d in self.df["diameter_mm"].tolist()]
			self.vocab = build_vocab(texts)
		else:
			self.vocab = vocab

	def __len__(self):
		"""__len__ function returns the number of samples in dataset

		returns:
			number of ROI samples in the dataset
		"""
		return len(self.df)

	def load_and_normalize_dicom(self, dicom_path: str) -> np.ndarray:
		"""load_and_normalize_dicom function loads dicom file and normalizes img array between 5th and 95th percentiles

		args:
			dicom_path: path to dicom file

		returns:
			normalized img array with values in [0, 1]
		"""
		dcm_dataset = dcmread(dicom_path)
		img_arr = dcm_dataset.pixel_array.astype(np.float32)
		vmin, vmax = np.percentile(img_arr, [5, 95])
		img_arr = np.clip((img_arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
		return img_arr

	def __getitem__(self, idx: int):
		"""__getitem__ function returns dataset sample at given index

		args:
			idx: sample index

		returns:
			tuple of (img_tensor, text_ids, text_length)
		"""
		row = self.df.iloc[idx]
		img = self.load_and_normalize_dicom(row["dicom_path"])
		H, W = img.shape
		points = ast.literal_eval(row["points_json"]) if isinstance(row["points_json"], str) else row["points_json"]
		xs = [p[0] for p in points]
		ys = [p[1] for p in points]
		xmin, ymin, xmax, ymax = max(0,min(xs)-16), max(0,min(ys)-16), min(W-1,max(xs)+16), min(H-1,max(ys)+16)
		crop = img[ymin:ymax+1, xmin:xmax+1]
		crop = sk_resize(crop, self.img_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

		text = sample_text_for_diameter(float(row["diameter_mm"]), rand=True)
		ids, length = tokenize(text, self.vocab, self.max_len)
		return torch.from_numpy(crop[None,...]), ids, length
