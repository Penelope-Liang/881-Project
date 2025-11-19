import ast
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from skimage.transform import resize as sk_resize
from typing import Optional, Tuple


def patient_from_path(path: str) -> Optional[str]:
	"""patient_from_path function extracts patient id from file path

	args:
		path: file path containing patient id

	returns:
		patient id str like "LIDC-IDRI-0001" or None
	"""
	match = re.search(r"(LIDC-IDRI-\d{4})", path.replace("\\","/"))
	return match.group(1) if match else None


def bin_label(diameter_mm: float) -> int:
	"""bin_label function bins diameter into categorical labels

	args:
		diameter_mm: diameter

	returns:
		bin label, like 0: <=6mm, 1: 6-10mm, 2: >10mm
	"""
	if diameter_mm <= 6.0:
		return 0
	elif diameter_mm <= 10.0:
		return 1
	return 2


class RoiRegressionDataset(Dataset):
	def __init__(self, roi_csv: str, patients_file: Optional[str], img_size: Tuple[int,int]=(256,256), pad: int = 16):
		"""RoiRegressionDataset constructor initializes dataset for regression training

		args:
			roi_csv: path to CSV file containing ROI data
			patients_file: path to file containing patient ids
			img_size: target size for resized imgs
			pad: pixels to expand around ROI bounding box
		"""
		self.df = pd.read_csv(roi_csv)
		if patients_file:
			patient_set = set([line.strip() for line in open(patients_file,"r",encoding="utf-8") if line.strip()])
			self.df = self.df[self.df["dicom_path"].astype(str).apply(lambda p: patient_from_path(p) in patient_set)]
		self.df = self.df.dropna(subset=["dicom_path","points_json","diameter_mm"]).reset_index(drop=True)
		self.img_size = img_size
		self.pad = pad

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
			tuple of img_tensor, diameter, class_label, dicom_path, bbox
		"""
		row = self.df.iloc[idx]
		img = self.load_and_normalize_dicom(row["dicom_path"])
		H, W = img.shape
		points = ast.literal_eval(row["points_json"]) if isinstance(row["points_json"], str) else row["points_json"]
		xs = [p[0] for p in points]
		ys = [p[1] for p in points]
		xmin, ymin, xmax, ymax = max(0,min(xs)-self.pad), max(0,min(ys)-self.pad), min(W-1,max(xs)+self.pad), min(H-1,max(ys)+self.pad)
		crop = img[ymin:ymax+1, xmin:xmax+1]
		crop = sk_resize(crop, self.img_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
		diameter_mm = float(row["diameter_mm"])
		cls = bin_label(diameter_mm)
		bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
		return torch.from_numpy(crop[None,...]), torch.tensor(diameter_mm, dtype=torch.float32), torch.tensor(cls, dtype=torch.long), row["dicom_path"], bbox
