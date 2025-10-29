import ast
import re
from typing import Tuple, Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from skimage.draw import polygon as sk_polygon
from skimage.transform import resize as sk_resize


def _patient_from_path(path: str) -> Optional[str]:
	m = re.search(r"(LIDC-IDRI-\d{4})", path.replace('\\','/'))
	return m.group(1) if m else None


def _load_patient_set(pfile: Optional[str]) -> Optional[Set[str]]:
	if not pfile:
		return None
	s: Set[str] = set()
	with open(pfile, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line:
				s.add(line)
	return s


class Roidataset(Dataset):
	def __init__(self, roi_csv: str, image_size: Tuple[int, int] = (256, 256), expand: int = 16, limit: Optional[int] = None, seed: int = 42, patients_file: Optional[str] = None):
		import pandas as pd
		self.df = pd.read_csv(roi_csv)
		self.df = self.df.dropna(subset=['dicom_path', 'points_json'])
		# optional patient filter
		pset = _load_patient_set(patients_file)
		if pset:
			self.df = self.df[self.df['dicom_path'].astype(str).apply(lambda p: _patient_from_path(p) in pset)]
		if limit is not None and limit > 0:
			self.df = self.df.sample(n=min(limit, len(self.df)), random_state=seed)
		self.image_size = image_size
		self.expand = expand

	def __len__(self) -> int:
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
		pts = ast.literal_eval(row['points_json']) if isinstance(row['points_json'], str) else row['points_json']
		xs = [p[0] for p in pts]
		ys = [p[1] for p in pts]
		xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
		xmin = max(0, xmin - self.expand)
		ymin = max(0, ymin - self.expand)
		xmax = min(img.shape[1] - 1, xmax + self.expand)
		ymax = min(img.shape[0] - 1, ymax + self.expand)
		crop = img[ymin:ymax+1, xmin:xmax+1]
		mask = np.zeros_like(img, dtype=np.float32)
		r = np.array([p[1] for p in pts])
		c = np.array([p[0] for p in pts])
		rr, cc = sk_polygon(r, c, shape=img.shape)
		mask[rr, cc] = 1.0
		mask = mask[ymin:ymax+1, xmin:xmax+1]
		crop = sk_resize(crop, self.image_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
		mask = sk_resize(mask, self.image_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
		return torch.from_numpy(crop[None, ...]), torch.from_numpy(mask[None, ...])






