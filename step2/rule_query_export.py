import argparse
import os
import ast
import numpy as np
import pandas as pd
from pydicom import dcmread
from PIL import Image, ImageDraw
from skimage.draw import polygon as sk_polygon
import re
from typing import Tuple, List

def ensure_dir(path: str) -> None:
	"""ensure_dir function ensures directory exist

	args:
		path: directory path
	"""
	os.makedirs(path, exist_ok=True)


def parse_query(query: str) -> Tuple[str, float]:
	"""parse_query function parses query string into operator and threshold value

	args:
		query: query like ">=3mm"

	returns:
		tuple of operator, threshold
	"""
	query_normalized = query.strip().lower().replace(" ", "")
	match = re.match(r'(>=|<=|>|<|=)?([0-9]+\.?[0-9]*)mm', query_normalized)
	if not match:
		raise ValueError(f"Invalid query: {query}")
	operator = match.group(1) or ">="
	threshold = float(match.group(2))
	return operator, threshold


def filter_topk(df: pd.DataFrame, operator: str, threshold: float, k: int) -> pd.DataFrame:
	"""filter_topk function filters DataFrame by operator and threshold, return top-k rows sorted by diameter

	args:
		df: input DataFrame with max_roi_diameter_mm column
		operator: comparison operator
		threshold: threshold value in mm
		k: number of top rows to return

	returns:
		filtered DataFrame with top-k rows sorted by max_roi_diameter_mm descending
	"""
	if operator == ">=":
		df_filtered = df[df["max_roi_diameter_mm"] >= threshold]
	elif operator == ">":
		df_filtered = df[df["max_roi_diameter_mm"] > threshold]
	elif operator == "<=":
		df_filtered = df[df["max_roi_diameter_mm"] <= threshold]
	elif operator == "<":
		df_filtered = df[df["max_roi_diameter_mm"] < threshold]
	elif operator == "=":
		df_filtered = df[np.isclose(df["max_roi_diameter_mm"], threshold, atol=0.1)]
	else:
		raise ValueError(f"Unknown operator: {operator}")
	return df_filtered.sort_values("max_roi_diameter_mm", ascending=False).head(k)


def draw_overlay(dicom_path: str, points: List[List[int]], output_png: str) -> None:
	"""draw_overlay function draws ROI polygon overlay on dcm img and save as PNG

	args:
		dicom_path: path to dcm file
		points: list of (x, y) for ROI
		output_png: path to output PNG file
	"""
	dcm_dataset = dcmread(dicom_path)
	img_arr = dcm_dataset.pixel_array.astype(np.float32)
	vmin, vmax = np.percentile(img_arr, [5, 95])
	img_arr = np.clip((img_arr - vmin) / (vmax - vmin + 1e-6) * 255.0, 0, 255).astype(np.uint8)
	img = Image.fromarray(img_arr).convert("RGB")
	draw = ImageDraw.Draw(img)
	if points:
		points_int = [(int(x), int(y)) for x, y in points]
		points_closed = points_int + [points_int[0]]
		draw.line(points_closed, fill=(255, 0, 0), width=2)
	img.save(output_png)


def save_mask(dicom_path: str, points: List[List[int]], output_png: str) -> None:
	"""save_mask function creates binary mask from ROI polygon and save as PNG

	args:
		dicom_path: path to dcm file
		points: list of (x, y) for ROI
		output_png: path to output PNG file
	"""
	dcm_dataset = dcmread(dicom_path)
	shape = (int(dcm_dataset.Rows), int(dcm_dataset.Columns))
	mask = np.zeros(shape, dtype=np.uint8)
	if points:
		r = np.array([point[1] for point in points])
		c = np.array([point[0] for point in points])
		rr, cc = sk_polygon(r, c, shape=shape)
		mask[rr, cc] = 255
	Image.fromarray(mask).save(output_png)


def main():
	"""main function is for query and export top-k slices with overlays and masks,
	loads slice and ROI CSV files, filters slices by query condition,
	selects top-k slices, and exports overlay PNGs and mask PNGs for each hit.
	"""
	parser = argparse.ArgumentParser(description='Rule-based query and export top-k slices with overlays/masks')
	parser.add_argument('--index', required=True, help='slice_index_with_diam.csv')
	parser.add_argument('--roi-csv', required=True, help='roi_with_diam.csv (to fetch ROI points)')
	parser.add_argument('--query', required=True, help='e.g., ">=3mm"')
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--out', dest='out_dir', required=True)
	args = parser.parse_args()

	ensure_dir(args.out_dir)
	print(f"[rule] loading {args.index} and {args.roi_csv}")
	df_slice = pd.read_csv(args.index)
	df_roi = pd.read_csv(args.roi_csv)
	operator, threshold = parse_query(args.query)
	print(f"[rule] query parsed: operator={operator}, threshold={threshold} mm")
	df_hit = filter_topk(df_slice, operator, threshold, args.topk).reset_index(drop=True)
	print(f"[rule] hits={len(df_hit)}")

	for i, row in df_hit.iterrows():
		sop = row["sop_uid"]
		candidates = df_roi[df_roi["sop_uid"] == sop].sort_values("diameter_mm", ascending=False)
		if candidates.empty:
			print(f"[rule:{i}] no ROI rows for sop={sop}")
			continue
		r = candidates.iloc[0]
		points = ast.literal_eval(r["points_json"]) if isinstance(r["points_json"], str) else r["points_json"]
		dicom_path = row["dicom_path"]
		base = os.path.join(args.out_dir, f"hit_{i:03d}_diameter_{row['max_roi_diameter_mm']:.1f}mm")
		# copy dcm not necessary; we just write overlays/masks alongside
		try:
			draw_overlay(dicom_path, points, base + "_overlay.png")
			save_mask(dicom_path, points, base + "_mask.png")
			print(f"[rule:{i}] wrote {base}_overlay.png and _mask.png")
		except Exception as e:
			print(f"[rule:{i}] export failed: {e}")

	df_hit.to_csv(os.path.join(args.out_dir, "topk_hits.csv"), index=False)
	print(f"[rule] wrote hit list: {os.path.join(args.out_dir, 'topk_hits.csv')}")
	print("[rule] done")


if __name__ == '__main__':
	main()
