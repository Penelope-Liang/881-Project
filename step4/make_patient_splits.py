import argparse
import os
import re
import random
import pandas as pd


def patient_from_path(p: str) -> str:
	m = re.search(r"(LIDC-IDRI-\d{4})", p.replace('\\','/'))
	return m.group(1) if m else 'UNKNOWN'


def write_list(path: str, items):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		for x in items:
			f.write(str(x) + '\n')


def main():
	ap = argparse.ArgumentParser(description='Make patient-level splits from ROI CSV')
	ap.add_argument('--roi-csv', required=True)
	ap.add_argument('--out-dir', required=True)
	ap.add_argument('--train', type=float, default=0.7)
	ap.add_argument('--val', type=float, default=0.1)
	ap.add_argument('--test', type=float, default=0.2)
	ap.add_argument('--seed', type=int, default=42)
	args = ap.parse_args()

	df = pd.read_csv(args.roi_csv)
	df['patient'] = df['dicom_path'].astype(str).apply(patient_from_path)
	patients = sorted(set(df['patient']) - {'UNKNOWN'})
	random.Random(args.seed).shuffle(patients)
	n = len(patients)
	nt = int(n * args.train)
	nv = int(n * args.val)
	train_pat = patients[:nt]
	val_pat = patients[nt:nt+nv]
	test_pat = patients[nt+nv:]

	os.makedirs(args.out_dir, exist_ok=True)
	write_list(os.path.join(args.out_dir, 'train_patients.txt'), train_pat)
	write_list(os.path.join(args.out_dir, 'val_patients.txt'), val_pat)
	write_list(os.path.join(args.out_dir, 'test_patients.txt'), test_pat)
	print(f"[splits] patients: total={n} train={len(train_pat)} val={len(val_pat)} test={len(test_pat)}")
	print(f"[splits] wrote {args.out_dir}")


if __name__ == '__main__':
	main()
