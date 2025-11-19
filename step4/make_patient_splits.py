import argparse
import os
import re
import random
import pandas as pd
from typing import List


def patient_from_path(path: str) -> str:
	"""patient_from_path function extracts patient id from file path

	args:
		path: file path containing patient id

	returns:
		patient id str like "LIDC-IDRI-0001" or "UNKNOWN"
	"""
	match = re.search(r"(LIDC-IDRI-\d{4})", path.replace("\\","/"))
	return match.group(1) if match else "UNKNOWN"


def write_list(path: str, items: List[str]) -> None:
	"""write_list function writes list of items to text file

	args:
		path: output file path
		items: list of items to write
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		for x in items:
			f.write(str(x) + "\n")


def main() -> None:
	"""main function loads ROI CSV, extracts patient ids, shuffles patients, splits into train, validation, and test sets
	based on specified ratios, and writes patient id lists to text files.
	"""
	parser = argparse.ArgumentParser(description="Split patient into train, validation, and test sets")
	parser.add_argument("--roi-csv", required=True)
	parser.add_argument("--out", dest="out_dir", required=True)
	parser.add_argument("--train", type=float, default=0.7)
	parser.add_argument("--val", type=float, default=0.1)
	parser.add_argument("--test", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	df = pd.read_csv(args.roi_csv)
	df["patient"] = df["dicom_path"].astype(str).apply(patient_from_path)
	patients = sorted(set(df["patient"]) - {"UNKNOWN"})
	random.Random(args.seed).shuffle(patients)
	n = len(patients)
	num_train = int(n * args.train)
	num_validation = int(n * args.val)
	train_patients = patients[:num_train]
	validation_patients = patients[num_train:num_train+num_validation]
	test_patients = patients[num_train+num_validation:]

	os.makedirs(args.out_dir, exist_ok=True)
	write_list(os.path.join(args.out_dir, "train_patients.txt"), train_patients)
	write_list(os.path.join(args.out_dir, "validation_patients.txt"), validation_patients)
	write_list(os.path.join(args.out_dir, "test_patients.txt"), test_patients)
	print(f"[splits] patients: total={n} train={len(train_patients)} val={len(validation_patients)} test={len(test_patients)}")
	print(f"[splits] wrote {args.out_dir}")


if __name__ == "__main__":
	main()
