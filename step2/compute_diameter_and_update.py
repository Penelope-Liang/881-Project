import argparse
import os
import json
# import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def ensure_dir(path: str) -> None:
	"""ensure_dir function ensures directory exist

	args:
		path: directory path
	"""
	os.makedirs(path, exist_ok=True)


def compute_max_diameter_mm(points: List[Tuple[int, int]], pixel_spacing: Optional[Tuple[float, float]]) -> float:
	"""compute_max_diameter_mm function computes max diameter of ROI

	args:
		points: list of (x, y) coordinates for ROI polygon
		pixel_spacing: pixel spacing in (row, col), or None

	returns:
		max diameter, or 0.0
	"""
	if pixel_spacing is None or len(points) < 2:
		return 0.0
	arr = np.asarray(points, dtype=np.float64)
	dx = arr[:, None, 0] - arr[None, :, 0]
	dy = arr[:, None, 1] - arr[None, :, 1]
	d2 = dx * dx + dy * dy
	max_dist_px = float(np.sqrt(d2.max()))
	row_mm, col_mm = pixel_spacing
	px_to_mm = float((row_mm + col_mm) / 2.0)
	return max_dist_px * px_to_mm


def compute_centroid(points: List[Tuple[int, int]]) -> Tuple[float, float]:
	"""compute_centroid function computes centroid of ROI

	args:
		points: list of (x, y) for ROI

	returns:
		tuple of centroid_x, centroid_y
	"""
	arr = np.asarray(points, dtype=np.float64)
	return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def compute_bbox(points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
	"""compute_bbox function computes bounding box of ROI polygon

	args:
		points: list of (x, y) for ROI

	returns:
		tuple of xmin, ymin, xmax, ymax bounding box coordinates
	"""
	xs = [p[0] for p in points]
	ys = [p[1] for p in points]
	return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def main():
	"""main function loads slice_index.json and roi_index.json from input directory,
	computes max diameter for each ROI based on pixel spacing,
	aggregates max ROI diameter per slice, and writes two CSV files:
	roi_with_diam.csv and slice_index_with_diam.csv.
	"""
	parser = argparse.ArgumentParser(description='Compute per-ROI diameter (mm) and aggregate per-slice; write CSVs')
	parser.add_argument('--in', dest='in_dir', required=True, help='Folder containing slice_index.json and roi_index.json')
	parser.add_argument('--out', dest='out_dir', required=True, help='Output folder for CSVs')
	args = parser.parse_args()

	print(f"[diameter] loading indexes from {args.in_dir}")
	slice_idx_path = os.path.join(args.in_dir, "slice_index.json")
	roi_idx_path = os.path.join(args.in_dir, "roi_index.json")
	assert os.path.exists(slice_idx_path), f"Missing {slice_idx_path}"
	assert os.path.exists(roi_idx_path), f"Missing {roi_idx_path}"

	with open(slice_idx_path, "r", encoding="utf-8") as f:
		slice_idx: List[Dict] = json.load(f)
	with open(roi_idx_path, "r", encoding="utf-8") as f:
		roi_idx: List[Dict] = json.load(f)

	print(f"[diameter] Slice records={len(slice_idx)}, roi records={len(roi_idx)}")

	sop_to_meta: Dict[str, Dict] = {s["sop_uid"]: s for s in slice_idx}

	roi_rows: List[Dict] = []
	for r in roi_idx:
		sop = r["sop_uid"]
		meta = sop_to_meta.get(sop)
		if meta is None:
			continue
		pixel_spacing = None
		if meta.get("pixel_spacing") and len(meta["pixel_spacing"]) == 2:
			pixel_spacing = (float(meta["pixel_spacing"][0]), float(meta["pixel_spacing"][1]))
		points = r["points"]
		diameter_mm = compute_max_diameter_mm(points, pixel_spacing)
		cx, cy = compute_centroid(points)
		xmin, ymin, xmax, ymax = compute_bbox(points)
		roi_rows.append({
			"series_uid": meta["series_uid"],
			"sop_uid": sop,
			"dicom_path": meta["dicom_path"],
			"pixel_spacing_row": pixel_spacing[0] if pixel_spacing else None,
			"pixel_spacing_col": pixel_spacing[1] if pixel_spacing else None,
			"diameter_mm": diameter_mm,
			"centroid_x": cx,
			"centroid_y": cy,
			"bbox_xmin": xmin,
			"bbox_ymin": ymin,
			"bbox_xmax": xmax,
			"bbox_ymax": ymax,
			"points_len": len(points),
			"points_json": json.dumps(points),
		})

	roi_df = pd.DataFrame.from_records(roi_rows)
	ensure_dir(args.out_dir)
	roi_csv = os.path.join(args.out_dir, "roi_with_diam.csv")
	roi_df.to_csv(roi_csv, index=False)
	print(f"[diameter] Wrote ROI CSV: {roi_csv} rows={len(roi_df)}")

	slice_rows: List[Dict] = []
	if not roi_df.empty:
		grp = roi_df.groupby("sop_uid")["diameter_mm"].max().rename("max_roi_diameter_mm").reset_index()
		sop_to_max = {row["sop_uid"]: row["max_roi_diameter_mm"] for _, row in grp.iterrows()}
		for s in slice_idx:
			m = sop_to_max.get(s["sop_uid"], 0.0)
			slice_rows.append({
				"xml_path": s["xml_path"],
				"series_uid": s["series_uid"],
				"sop_uid": s["sop_uid"],
				"dicom_path": s["dicom_path"],
				"pixel_spacing_row": (s["pixel_spacing"][0] if s.get("pixel_spacing") else None),
				"pixel_spacing_col": (s["pixel_spacing"][1] if s.get("pixel_spacing") else None),
				"num_rois": s["num_rois"],
				"max_roi_diameter_mm": m,
			})
	else:
		for s in slice_idx:
			slice_rows.append({
				"xml_path": s["xml_path"],
				"series_uid": s["series_uid"],
				"sop_uid": s["sop_uid"],
				"dicom_path": s["dicom_path"],
				"pixel_spacing_row": (s["pixel_spacing"][0] if s.get("pixel_spacing") else None),
				"pixel_spacing_col": (s["pixel_spacing"][1] if s.get("pixel_spacing") else None),
				"num_rois": s["num_rois"],
				"max_roi_diameter_mm": 0.0,
			})

	slice_df = pd.DataFrame.from_records(slice_rows)
	slice_csv = os.path.join(args.out_dir, "slice_index_with_diam.csv")
	slice_df.to_csv(slice_csv, index=False)
	print(f"[diameter] Wrote slice CSV: {slice_csv} rows={len(slice_df)}")

	print("[diameter] Done")


if __name__ == '__main__':
	main()
