import argparse
import os
import glob
import json
from typing import List, Tuple, Dict

import numpy as np
from lxml import etree
from pydicom import dcmread
from PIL import Image
from skimage.draw import polygon as sk_polygon

LIDC_NS = {"lidc": "http://www.nih.gov"}


def ensure_dir(p: str) -> None:
	os.makedirs(p, exist_ok=True)


def parse_xml(xml_path: str) -> Tuple[str, List[Dict]]:
	t = etree.parse(xml_path)
	root = t.getroot()
	su = None
	n = root.find('.//lidc:SeriesInstanceUid', namespaces=LIDC_NS)
	if n is None:
		n = root.find('.//lidc:SeriesInstanceUID', namespaces=LIDC_NS)
	if n is not None and n.text:
		su = n.text
	rois: List[Dict] = []
	for un in root.findall('.//lidc:unblindedReadNodule', namespaces=LIDC_NS):
		for r in un.findall('./lidc:roi', namespaces=LIDC_NS):
			sn = r.find('./lidc:imageSOP_UID', namespaces=LIDC_NS)
			sop = sn.text if sn is not None else None
			pts = []
			for em in r.findall('./lidc:edgeMap', namespaces=LIDC_NS):
				xn = em.find('./lidc:xCoord', namespaces=LIDC_NS)
				yn = em.find('./lidc:yCoord', namespaces=LIDC_NS)
				if xn is not None and yn is not None and xn.text and yn.text:
					pts.append((int(xn.text), int(yn.text)))
			if sop and len(pts) >= 3:
				rois.append({"sop_uid": sop, "points": pts})
	return su, rois


def build_sop_map(series_dir: str) -> Dict[str, str]:
	m: Dict[str, str] = {}
	for p in glob.iglob(os.path.join(series_dir, '**', '*.dcm'), recursive=True):
		try:
			ds = dcmread(p, stop_before_pixels=True, force=True)
			sop = getattr(ds, 'SOPInstanceUID', None)
			if sop:
				m[sop] = p
		except Exception:
			continue
	return m


def make_slice_mask(shape: Tuple[int, int], roi_list: List[List[Tuple[int, int]]]) -> np.ndarray:
	mask = np.zeros(shape, dtype=np.uint8)
	for pts in roi_list:
		r = np.array([p[1] for p in pts])
		c = np.array([p[0] for p in pts])
		rr, cc = sk_polygon(r, c, shape=shape)
		mask[rr, cc] = 255
	return mask


def write_png(arr: np.ndarray, path: str) -> None:
	Image.fromarray(arr).save(path)


def main():
	parser = argparse.ArgumentParser(description='Step2: Build slice-level and ROI-level indexes with masks and previews')
	parser.add_argument('--data-root', required=True)
	parser.add_argument('--out', required=True, help='Output folder for indexes and previews')
	parser.add_argument('--preview', type=int, default=10, help='Max number of slice previews to write')
	args = parser.parse_args()

	print(f"[step2] building indexes from {args.data_root}")
	ensure_dir(args.out)
	ensure_dir(os.path.join(args.out, 'previews'))

	xml_files = list(glob.iglob(os.path.join(args.data_root, '**', '*.xml'), recursive=True))
	print(f"[step2] found xml files: {len(xml_files)}")

	slice_index: List[Dict] = []
	roi_index: List[Dict] = []
	preview_written = 0

	for xml in xml_files:
		series_uid, rois = None, []
		try:
			series_uid, rois = parse_xml(xml)
		except Exception as e:
			print(f"[step2] parse_xml failed: {xml} err={e}")
			continue
		if not series_uid or not rois:
			continue
		series_dir = os.path.dirname(xml)
		sop_map = build_sop_map(series_dir)
		# group by sop
		sop_to_pts: Dict[str, List[List[Tuple[int, int]]]] = {}
		for r in rois:
			p = sop_map.get(r['sop_uid'])
			if not p:
				continue
			sop_to_pts.setdefault(r['sop_uid'], []).append(r['points'])
		for sop, all_pts in sop_to_pts.items():
			dcm_path = sop_map[sop]
			try:
				ds = dcmread(dcm_path)
			except Exception as e:
				print(f"[step2] read dicom failed: {dcm_path} err={e}")
				continue
			img = ds.pixel_array.astype(np.uint8)
			shape = (int(ds.Rows), int(ds.Columns))
			mask = make_slice_mask(shape, all_pts)
			slice_item = {
				'xml_path': xml,
				'series_uid': series_uid,
				'sop_uid': sop,
				'dicom_path': dcm_path,
				'pixel_spacing': [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])] if hasattr(ds, 'PixelSpacing') and len(ds.PixelSpacing) == 2 else None,
				'num_rois': len(all_pts),
			}
			slice_index.append(slice_item)
			# roi index entries
			for pts in all_pts:
				roi_index.append({
					'xml_path': xml,
					'series_uid': series_uid,
					'sop_uid': sop,
					'dicom_path': dcm_path,
					'points': pts,
				})
			# limited previews
			if preview_written < args.preview:
				# write image and mask pngs side-by-side (two files)
				img_norm = ds.pixel_array.astype(np.float32)
				vmin, vmax = np.percentile(img_norm, [5, 95])
				img_norm = np.clip((img_norm - vmin) / (vmax - vmin + 1e-6) * 255.0, 0, 255).astype(np.uint8)
				write_png(img_norm, os.path.join(args.out, 'previews', f'slice_{preview_written:03d}_img.png'))
				write_png(mask, os.path.join(args.out, 'previews', f'slice_{preview_written:03d}_mask.png'))
				preview_written += 1

	# write indexes
	with open(os.path.join(args.out, 'slice_index.json'), 'w', encoding='utf-8') as f:
		json.dump(slice_index, f, indent=2)
	with open(os.path.join(args.out, 'roi_index.json'), 'w', encoding='utf-8') as f:
		json.dump(roi_index, f, indent=2)

	print(f"[step2] slice_index records={len(slice_index)}, roi_index records={len(roi_index)}")
	print(f"[step2] previews written={preview_written}, out={os.path.join(args.out, 'previews')}")


if __name__ == '__main__':
	main()
