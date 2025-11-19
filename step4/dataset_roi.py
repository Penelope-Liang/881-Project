import ast
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from skimage.draw import polygon as sk_polygon
from skimage.transform import resize as sk_resize
import pandas as pd
from typing import Tuple, Optional, Set

def patient_from_path(path: str) -> Optional[str]:
	"""patient_from_path function extracts patient id from file path

	args:
		path: file path containing patient id

	returns:
		patient id str like "LIDC-IDRI-0001" or None
	"""
	match = re.search(r"(LIDC-IDRI-\d{4})", path.replace("\\","/"))
	return match.group(1) if match else None


def load_patient_set(patient_file: Optional[str]) -> Optional[Set[str]]:
	"""load_patient_set function loads patient ids from file into a set

	args:
		patient_file: path to file containing patient ids

	returns:
		set of patient id str, or None
	"""
	if not patient_file:
		return None
	patient_set: Set[str] = set()
	with open(patient_file, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				patient_set.add(line)
	return patient_set


class Roidataset(Dataset):
	def __init__(self, roi_csv: str, img_size: Tuple[int, int] = (256, 256), pad: int = 16, limit: Optional[int] = None, seed: int = 42, patients_file: Optional[str] = None):
		"""__init__ function initializes ROI dataset

		args:
			roi_csv: path to CSV file containing ROI data
			img_size: target img size for resizing
			pad: padding pixels to add around ROI bounding box
			limit: max number of samples
			seed: random seed for sampling
			patients_file: path to file containing patient ids
		"""
		self.df = pd.read_csv(roi_csv)
		self.df = self.df.dropna(subset=["dicom_path", "points_json"])
		patient_set = load_patient_set(patients_file)
		if patient_set:
			self.df = self.df[self.df["dicom_path"].astype(str).apply(lambda path: patient_from_path(path) in patient_set)]
		if limit is not None and limit > 0:
			self.df = self.df.sample(n=min(limit, len(self.df)), random_state=seed)
		self.img_size = img_size
		self.pad = pad

	def __len__(self) -> int:
		"""__len__ function returns the number of samples in the dataset

		returns:
			number of samples in dataset
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
		"""__getitem__ function returns the sample at the given index

		args:
			idx: sample index

		returns:
			tuple of crop_tensor, mask_tensor with shape 1, height, width
		"""
		row = self.df.iloc[idx]
		img = self.load_and_normalize_dicom(row["dicom_path"])
		points = ast.literal_eval(row["points_json"]) if isinstance(row["points_json"], str) else row["points_json"]
		xs = [p[0] for p in points]
		ys = [p[1] for p in points]
		xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
		xmin = max(0, xmin - self.pad)
		ymin = max(0, ymin - self.pad)
		xmax = min(img.shape[1] - 1, xmax + self.pad)
		ymax = min(img.shape[0] - 1, ymax + self.pad)
		crop = img[ymin:ymax+1, xmin:xmax+1]
		mask = np.zeros_like(img, dtype=np.float32)
		r = np.array([p[1] for p in points])
		c = np.array([p[0] for p in points])
		rr, cc = sk_polygon(r, c, shape=img.shape)
		mask[rr, cc] = 1.0
		mask = mask[ymin:ymax+1, xmin:xmax+1]
		crop = sk_resize(crop, self.img_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
		mask = sk_resize(mask, self.img_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
		return torch.from_numpy(crop[None, ...]), torch.from_numpy(mask[None, ...])
