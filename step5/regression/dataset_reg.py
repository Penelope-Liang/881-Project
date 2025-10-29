import ast
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from skimage.transform import resize as sk_resize


def _patient_from_path(path: str) -> Optional[str]:
	m = re.search(r"(LIDC-IDRI-\d{4})", path.replace('\\','/'))
	return m.group(1) if m else None


def bin_label(d_mm: float) -> int:
	if d_mm <= 6.0:
		return 0
	elif d_mm <= 10.0:
		return 1
	return 2


class RoiRegressionDataset(Dataset):
	def __init__(self, roi_csv: str, patients_file: Optional[str], image_size: Tuple[int,int]=(256,256), expand: int = 16):
		self.df = pd.read_csv(roi_csv)
		if patients_file:
			ps = set([l.strip() for l in open(patients_file,'r',encoding='utf-8') if l.strip()])
			self.df = self.df[self.df['dicom_path'].astype(str).apply(lambda p: _patient_from_path(p) in ps)]
		self.df = self.df.dropna(subset=['dicom_path','points_json','diameter_mm']).reset_index(drop=True)
		self.image_size = image_size
		self.expand = expand

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
		xmin, ymin, xmax, ymax = max(0,min(xs)-self.expand), max(0,min(ys)-self.expand), min(W-1,max(xs)+self.expand), min(H-1,max(ys)+self.expand)
		crop = img[ymin:ymax+1, xmin:xmax+1]
		crop = sk_resize(crop, self.image_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
		diam = float(row['diameter_mm'])
		cls = bin_label(diam)
		bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
		return torch.from_numpy(crop[None,...]), torch.tensor(diam, dtype=torch.float32), torch.tensor(cls, dtype=torch.long), row['dicom_path'], bbox
