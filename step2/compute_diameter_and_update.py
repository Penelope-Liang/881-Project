import argparse
import os
import json
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


def ensure_dir(p: str) -> None:
	os.makedirs(p, exist_ok=True)


def compute_max_diameter_mm(points: List[Tuple[int, int]], pixel_spacing: Optional[Tuple[float, float]]) -> float:
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
	arr = np.asarray(points, dtype=np.float64)
	return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def compute_bbox(points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
	xs = [p[0] for p in points]
	ys = [p[1] for p in points]
	return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def main():
	parser = argparse.ArgumentParser(description='Compute per-ROI diameter (mm) and aggregate per-slice; write CSVs')
	parser.add_argument('--in', dest='in_dir', required=True, help='Folder containing slice_index.json and roi_index.json')
	parser.add_argument('--out', dest='out_dir', required=True, help='Output folder for CSVs')
	args = parser.parse_args()

	print(f"[diam] loading indexes from {args.in_dir}")
	slice_p = os.path.join(args.in_dir, 'slice_index.json')
	roi_p = os.path.join(args.in_dir, 'roi_index.json')
	assert os.path.exists(slice_p), f"Missing {slice_p}"
	assert os.path.exists(roi_p), f"Missing {roi_p}"

	with open(slice_p, 'r', encoding='utf-8') as f:
		slice_idx: List[Dict] = json.load(f)
	with open(roi_p, 'r', encoding='utf-8') as f:
		roi_idx: List[Dict] = json.load(f)

	print(f"[diam] slice records={len(slice_idx)}, roi records={len(roi_idx)}")

	# Build map from sop_uid to slice meta
	sop_to_meta: Dict[str, Dict] = {s['sop_uid']: s for s in slice_idx}

	# Build ROI table
	roi_rows: List[Dict] = []
	for r in roi_idx:
		sop = r['sop_uid']
		meta = sop_to_meta.get(sop)
		if meta is None:
			continue
		pixel_spacing = None
		if meta.get('pixel_spacing') and len(meta['pixel_spacing']) == 2:
			pixel_spacing = (float(meta['pixel_spacing'][0]), float(meta['pixel_spacing'][1]))
		pts = r['points']
		diam_mm = compute_max_diameter_mm(pts, pixel_spacing)
		cx, cy = compute_centroid(pts)
		xmin, ymin, xmax, ymax = compute_bbox(pts)
		roi_rows.append({
			'series_uid': meta['series_uid'],
			'sop_uid': sop,
			'dicom_path': meta['dicom_path'],
			'pixel_spacing_row': pixel_spacing[0] if pixel_spacing else None,
			'pixel_spacing_col': pixel_spacing[1] if pixel_spacing else None,
			'diameter_mm': diam_mm,
			'centroid_x': cx,
			'centroid_y': cy,
			'bbox_xmin': xmin,
			'bbox_ymin': ymin,
			'bbox_xmax': xmax,
			'bbox_ymax': ymax,
			'points_len': len(pts),
			'points_json': json.dumps(pts),
		})

	roi_df = pd.DataFrame.from_records(roi_rows)
	ensure_dir(args.out_dir)
	roi_csv = os.path.join(args.out_dir, 'roi_with_diam.csv')
	roi_df.to_csv(roi_csv, index=False)
	print(f"[diam] wrote ROI CSV: {roi_csv} rows={len(roi_df)}")

	# Aggregate per slice: max diameter among ROIs
	slice_rows: List[Dict] = []
	if not roi_df.empty:
		grp = roi_df.groupby('sop_uid')['diameter_mm'].max().rename('max_roi_diameter_mm').reset_index()
		sop_to_max = {row['sop_uid']: row['max_roi_diameter_mm'] for _, row in grp.iterrows()}
		for s in slice_idx:
			m = sop_to_max.get(s['sop_uid'], 0.0)
			slice_rows.append({
				'xml_path': s['xml_path'],
				'series_uid': s['series_uid'],
				'sop_uid': s['sop_uid'],
				'dicom_path': s['dicom_path'],
				'pixel_spacing_row': (s['pixel_spacing'][0] if s.get('pixel_spacing') else None),
				'pixel_spacing_col': (s['pixel_spacing'][1] if s.get('pixel_spacing') else None),
				'num_rois': s['num_rois'],
				'max_roi_diameter_mm': m,
			})
	else:
		for s in slice_idx:
			slice_rows.append({
				'xml_path': s['xml_path'],
				'series_uid': s['series_uid'],
				'sop_uid': s['sop_uid'],
				'dicom_path': s['dicom_path'],
				'pixel_spacing_row': (s['pixel_spacing'][0] if s.get('pixel_spacing') else None),
				'pixel_spacing_col': (s['pixel_spacing'][1] if s.get('pixel_spacing') else None),
				'num_rois': s['num_rois'],
				'max_roi_diameter_mm': 0.0,
			})

	slice_df = pd.DataFrame.from_records(slice_rows)
	slice_csv = os.path.join(args.out_dir, 'slice_index_with_diam.csv')
	slice_df.to_csv(slice_csv, index=False)
	print(f"[diam] wrote slice CSV: {slice_csv} rows={len(slice_df)}")

	print("[diam] done")


if __name__ == '__main__':
	main()

