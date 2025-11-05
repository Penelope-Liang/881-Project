import argparse
import os
import ast
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
from pydicom import dcmread
from PIL import Image, ImageDraw
from skimage.draw import polygon as sk_polygon


def ensure_dir(p: str) -> None:
	os.makedirs(p, exist_ok=True)


def parse_query(q: str) -> Tuple[str, float]:
	s = q.strip().lower().replace(' ', '')
	import re
	m = re.match(r'(>=|<=|>|<|=)?([0-9]+\.?[0-9]*)mm', s)
	if not m:
		raise ValueError(f"Invalid query: {q}")
	op = m.group(1) or '>='
	val = float(m.group(2))
	return op, val


def filter_topk(df: pd.DataFrame, op: str, val: float, k: int) -> pd.DataFrame:
	if op == '>=':
		df2 = df[df['max_roi_diameter_mm'] >= val]
	elif op == '>':
		df2 = df[df['max_roi_diameter_mm'] > val]
	elif op == '<=':
		df2 = df[df['max_roi_diameter_mm'] <= val]
	elif op == '<':
		df2 = df[df['max_roi_diameter_mm'] < val]
	elif op == '=':
		df2 = df[np.isclose(df['max_roi_diameter_mm'], val, atol=0.1)]
	else:
		raise ValueError(f"Unknown op: {op}")
	return df2.sort_values('max_roi_diameter_mm', ascending=False).head(k)


def draw_overlay(dicom_path: str, points: List[List[int]], out_png: str) -> None:
	ds = dcmread(dicom_path)
	arr = ds.pixel_array.astype(np.float32)
	vmin, vmax = np.percentile(arr, [5, 95])
	arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6) * 255.0, 0, 255).astype(np.uint8)
	img = Image.fromarray(arr).convert('RGB')
	draw = ImageDraw.Draw(img)
	if points:
		pts = [(int(x), int(y)) for x, y in points]
		pts2 = pts + [pts[0]]
		draw.line(pts2, fill=(255, 0, 0), width=2)
	img.save(out_png)


def save_mask(dicom_path: str, points: List[List[int]], out_png: str) -> None:
	ds = dcmread(dicom_path)
	shape = (int(ds.Rows), int(ds.Columns))
	mask = np.zeros(shape, dtype=np.uint8)
	if points:
		r = np.array([p[1] for p in points])
		c = np.array([p[0] for p in points])
		rr, cc = sk_polygon(r, c, shape=shape)
		mask[rr, cc] = 255
	Image.fromarray(mask).save(out_png)


def main():
	parser = argparse.ArgumentParser(description='Rule-based query and export top-k slices with overlays/masks')
	parser.add_argument('--index', required=True, help='slice_index_with_diam.csv')
	parser.add_argument('--roi-csv', required=True, help='roi_with_diam.csv (to fetch ROI points)')
	parser.add_argument('--query', required=True, help='e.g., ">=3mm"')
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--out', required=True)
	args = parser.parse_args()

	ensure_dir(args.out)
	print(f"[rule] loading {args.index} and {args.roi_csv}")
	df_slice = pd.read_csv(args.index)
	df_roi = pd.read_csv(args.roi_csv)
	op, val = parse_query(args.query)
	print(f"[rule] query parsed: op={op}, val={val} mm")
	df_hit = filter_topk(df_slice, op, val, args.topk).reset_index(drop=True)
	print(f"[rule] hits={len(df_hit)}")

	# for each hit, pick the ROI with maximum diameter on that slice
	for i, row in df_hit.iterrows():
		sop = row['sop_uid']
		cand = df_roi[df_roi['sop_uid'] == sop].sort_values('diameter_mm', ascending=False)
		if cand.empty:
			print(f"[rule:{i}] no ROI rows for sop={sop}")
			continue
		r = cand.iloc[0]
		pts = ast.literal_eval(r['points_json']) if isinstance(r['points_json'], str) else r['points_json']
		dcm = row['dicom_path']
		base = os.path.join(args.out, f"hit_{i:03d}_diam_{row['max_roi_diameter_mm']:.1f}mm")
		# copy dicom not necessary; we just write overlays/masks alongside
		try:
			draw_overlay(dcm, pts, base + '_overlay.png')
			save_mask(dcm, pts, base + '_mask.png')
			print(f"[rule:{i}] wrote {base}_overlay.png and _mask.png")
		except Exception as e:
			print(f"[rule:{i}] export failed: {e}")

	# also save the hit list
	df_hit.to_csv(os.path.join(args.out, 'topk_hits.csv'), index=False)
	print(f"[rule] wrote hit list: {os.path.join(args.out, 'topk_hits.csv')}")
	print("[rule] done")


if __name__ == '__main__':
	main()

