import argparse
import os
import json

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> None:
	os.makedirs(p, exist_ok=True)


def main():
	parser = argparse.ArgumentParser(description='Step3: Build diameter stats and QC report')
	parser.add_argument('--in', dest='in_dir', required=True, help='Folder with slice_index_with_diam.csv and roi_with_diam.csv')
	parser.add_argument('--out', dest='out_dir', required=True, help='Output folder for report and figures')
	args = parser.parse_args()

	ensure_dir(args.out_dir)
	slice_csv = os.path.join(args.in_dir, 'slice_index_with_diam.csv')
	roi_csv = os.path.join(args.in_dir, 'roi_with_diam.csv')
	assert os.path.exists(slice_csv)
	assert os.path.exists(roi_csv)

	df_s = pd.read_csv(slice_csv)
	df_r = pd.read_csv(roi_csv)

	# Stats
	stats = {
		'total_slices': int(len(df_s)),
		'total_rois': int(len(df_r)),
		'ge3_slices': int((df_s['max_roi_diameter_mm'] >= 3.0).sum()),
		'ge3_rois': int((df_r['diameter_mm'] >= 3.0).sum()),
	}

	# Diameter histogram (ROI level)
	fig_path = os.path.join(args.out_dir, 'diameter_hist.png')
	plt.figure(figsize=(7,4))
	df_r['diameter_mm'].clip(lower=0, upper=60).hist(bins=60)
	plt.xlabel('ROI max diameter (mm)')
	plt.ylabel('Count')
	plt.tight_layout()
	plt.savefig(fig_path)
	plt.close()

	# Anomalies
	anom = df_r[(df_r['diameter_mm'] <= 0) | df_r['diameter_mm'].isna()]
	anom_path = os.path.join(args.out_dir, 'anomalies.csv')
	anom.to_csv(anom_path, index=False)

	# Report
	rep_path = os.path.join(args.out_dir, 'REPORT.md')
	with open(rep_path, 'w', encoding='utf-8') as f:
		f.write('# Step 3 - QC Report\n\n')
		f.write('## Summary\n')
		for k, v in stats.items():
			f.write(f'- {k}: {v}\n')
		f.write('\n## Figures\n')
		f.write(f'- Diameter histogram: {fig_path}\n')
		f.write('\n## Anomalies\n')
		f.write(f'- Anomalies CSV: {anom_path} (rows={len(anom)})\n')

	print(f"[step3] wrote report: {rep_path}")
	print(f"[step3] wrote histogram: {fig_path}")
	print(f"[step3] wrote anomalies: {anom_path}")


if __name__ == '__main__':
	main()




