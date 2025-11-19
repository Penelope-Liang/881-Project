import argparse
import os
# import json

import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> None:
	"""ensure_dir function ensures directory exist

	args:
		path: directory path
	"""
	os.makedirs(path, exist_ok=True)


def main():
	"""
	main function builds diameter stats and quality control report from step2 slice_index_with_diam.csv and roi_with_diam.csv,
	and calculates the total slices and ROIs, the number of slices and ROIs with diameter >= 3 mm, create ROI diameter 
	histogram and save as PNG. find anomalies data with diameter <= 0 or NA and save as anomalies.csv, generate REPORT.md.
	"""
	parser = argparse.ArgumentParser(description="Step3: Build diameter statistics and quality control report")
	parser.add_argument("--in", dest="in_dir", required=True, help="Folder with slice_index_with_diam.csv and roi_with_diam.csv")
	parser.add_argument("--out", dest="out_dir", required=True, help="Output folder for report and figures")
	args = parser.parse_args()

	ensure_dir(args.out_dir)
	slice_csv = os.path.join(args.in_dir, "slice_index_with_diam.csv")
	roi_csv = os.path.join(args.in_dir, "roi_with_diam.csv")
	assert os.path.exists(slice_csv)
	assert os.path.exists(roi_csv)

	slice_df = pd.read_csv(slice_csv)
	roi_df = pd.read_csv(roi_csv)

	stats = {
		"total_slices": int(len(slice_df)),
		"total_rois": int(len(roi_df)),
		"ge3_slices": int((slice_df["max_roi_diameter_mm"] >= 3.0).sum()),
		"ge3_rois": int((roi_df["diameter_mm"] >= 3.0).sum()),
	}

	fig_path = os.path.join(args.out_dir, "diameter_histogram.png")
	plt.figure(figsize=(7,4))
	roi_df["diameter_mm"].clip(lower=0, upper=60).hist(bins=60)
	plt.xlabel("ROI max diameter (mm)")
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(fig_path)
	plt.close()

	anomalies = roi_df[(roi_df["diameter_mm"] <= 0) | roi_df["diameter_mm"].isna()]
	anomalies_path = os.path.join(args.out_dir, "anomalies.csv")
	anomalies.to_csv(anomalies_path, index=False)

	report_path = os.path.join(args.out_dir, "REPORT.md")
	with open(report_path, "w", encoding="utf-8") as f:
		f.write("# Step 3 - Quality Control Report\n\n")
		f.write("## Summary\n")
		for k, v in stats.items():
			f.write(f"- {k}: {v}\n")
		f.write("\n## Figures\n")
		f.write(f"- Diameter histogram: {fig_path}\n")
		f.write("\n## Anomalies\n")
		f.write(f"- Anomalies CSV: {anomalies_path} (rows={len(anomalies)})\n")

	print(f"[step3] Wrote report: {report_path}")
	print(f"[step3] Wrote histogram: {fig_path}")
	print(f"[step3] Wrote anomalies: {anomalies_path}")


if __name__ == "__main__":
	main()
