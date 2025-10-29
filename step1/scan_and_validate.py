import argparse
import os
import glob
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
from lxml import etree
from pydicom import dcmread
from PIL import Image, ImageDraw


LIDC_NS = {"lidc": "http://www.nih.gov"}


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def parse_lidc_xml(xml_path: str) -> Tuple[Optional[str], List[Dict]]:
	"""Parse SeriesInstanceUID and a list of ROI dicts: {sop_uid, points}.
	Returns (series_uid, rois)."""
	tree = etree.parse(xml_path)
	root = tree.getroot()
	series_uid = None
	# Handle two possible tag names
	node = root.find('.//lidc:SeriesInstanceUid', namespaces=LIDC_NS)
	if node is None:
		node = root.find('.//lidc:SeriesInstanceUID', namespaces=LIDC_NS)
	if node is not None and node.text:
		series_uid = node.text

	rois: List[Dict] = []
	for n in root.findall('.//lidc:unblindedReadNodule', namespaces=LIDC_NS):
		for roi in n.findall('./lidc:roi', namespaces=LIDC_NS):
			sop_node = roi.find('./lidc:imageSOP_UID', namespaces=LIDC_NS)
			sop_uid = sop_node.text if sop_node is not None else None
			points: List[Tuple[int, int]] = []
			for em in roi.findall('./lidc:edgeMap', namespaces=LIDC_NS):
				xn = em.find('./lidc:xCoord', namespaces=LIDC_NS)
				yn = em.find('./lidc:yCoord', namespaces=LIDC_NS)
				if xn is not None and yn is not None and xn.text and yn.text:
					points.append((int(xn.text), int(yn.text)))
			if sop_uid and len(points) >= 3:
				rois.append({"sop_uid": sop_uid, "points": points})
	return series_uid, rois


def build_sop_to_path(series_dir: str) -> Dict[str, str]:
	"""Scan DICOM files under series_dir (recursive) and build SOPInstanceUID -> file path map."""
	sop_map: Dict[str, str] = {}
	for path in glob.iglob(os.path.join(series_dir, '**', '*.dcm'), recursive=True):
		try:
			ds = dcmread(path, stop_before_pixels=True, force=True)
		except Exception:
			continue
		sop = getattr(ds, 'SOPInstanceUID', None)
		if sop:
			sop_map[sop] = path
	return sop_map


def load_image_normalized(dicom_path: str) -> np.ndarray:
	ds = dcmread(dicom_path)
	arr = ds.pixel_array.astype(np.float32)
	vmin, vmax = np.percentile(arr, [5, 95])
	arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
	return arr


def draw_overlay(dicom_path: str, points: List[Tuple[int, int]], out_png: str) -> None:
	arr = load_image_normalized(dicom_path)
	img = Image.fromarray((arr * 255).astype(np.uint8)).convert('RGB')
	draw = ImageDraw.Draw(img)
	pts = [(int(x), int(y)) for x, y in points]
	if pts:
		# close polygon
		pts2 = pts + [pts[0]]
		draw.line(pts2, fill=(255, 0, 0), width=2)
	img.save(out_png)


def count_patients(data_root: str) -> int:
	count = 0
	for name in os.listdir(data_root):
		p = os.path.join(data_root, name)
		if os.path.isdir(p) and name.startswith('LIDC-IDRI-'):
			count += 1
	return count


def count_dicom_and_ct(data_root: str) -> Tuple[int, int]:
	"""Return (num_dicom_all, num_ct_dicom)."""
	all_cnt = 0
	ct_cnt = 0
	for root, _, files in os.walk(data_root):
		for name in files:
			if not name.lower().endswith('.dcm'):
				continue
			all_cnt += 1
			path = os.path.join(root, name)
			try:
				ds = dcmread(path, stop_before_pixels=True, force=True)
				if getattr(ds, 'Modality', None) == 'CT':
					ct_cnt += 1
			except Exception:
				pass
	return all_cnt, ct_cnt


def main() -> None:
	parser = argparse.ArgumentParser(description='Scan LIDC data and validate XML-DICOM pairing with visual proofs')
	parser.add_argument('--data-root', required=True, help='Path to LIDC-IDRI root')
	parser.add_argument('--samples', type=int, default=5, help='Number of XML files to sample for overlay checks')
	parser.add_argument('--out', required=True, help='Output directory for summary and overlays')
	args = parser.parse_args()

	print(f"[scan] Starting scan_and_validate with data_root={args.data_root}, samples={args.samples}, out={args.out}")
	ensure_dir(args.out)

	# High-level counts
	print("[scan] Counting patients, XMLs and DICOM files ...")
	n_patients = count_patients(args.data_root)
	xml_files = list(glob.iglob(os.path.join(args.data_root, '**', '*.xml'), recursive=True))
	n_xml = len(xml_files)
	n_dicom_all, n_ct = count_dicom_and_ct(args.data_root)
	print(f"[scan] Found patients={n_patients}, xml={n_xml}, dicom(all/ct)={n_dicom_all}/{n_ct}")

	summary = {
		'num_patients': n_patients,
		'num_xml': n_xml,
		'num_dicom_all': n_dicom_all,
		'num_dicom_ct': n_ct,
		'samples': [],
	}
	missing_pairs = []

	# Sample XMLs for overlay
	sample_paths = xml_files[: args.samples]
	print(f"[scan] Sampled {len(sample_paths)} XML files for overlays")
	for i, xml_path in enumerate(sample_paths):
		print(f"[scan:{i}] XML: {xml_path}")
		series_uid, rois = None, []
		try:
			series_uid, rois = parse_lidc_xml(xml_path)
			print(f"[scan:{i}] series_uid={series_uid}, roi_count={len(rois) if rois else 0}")
		except Exception as e:
			print(f"[scan:{i}] XML parse failed: {e}")
		item = {
			'xml_path': xml_path,
			'series_uid': series_uid,
			'roi_count': len(rois) if rois else 0,
			'dicom_found': False,
			'overlay_path': None,
			'pixel_spacing': None,
		}
		if not rois or not series_uid:
			print(f"[scan:{i}] Skip: no_series_or_roi")
			summary['samples'].append(item)
			missing_pairs.append({'xml_path': xml_path, 'reason': 'no_series_or_roi'})
			continue

		series_dir = os.path.dirname(xml_path)
		sop_map = build_sop_to_path(series_dir)
		print(f"[scan:{i}] DICOM in series_dir={series_dir}: {len(sop_map)}")
		# pick first ROI that exists in DICOM set
		chosen = None
		for roi in rois:
			p = sop_map.get(roi['sop_uid'])
			if p:
				chosen = (roi, p)
				break
		if not chosen:
			print(f"[scan:{i}] SOP not found for any ROI in this series")
			summary['samples'].append(item)
			missing_pairs.append({'xml_path': xml_path, 'series_uid': series_uid, 'reason': 'sop_not_found'})
			continue

		roi, dcm_path = chosen
		print(f"[scan:{i}] Chosen SOP={roi['sop_uid']} dicom_path={dcm_path}")
		# pixel spacing
		ps = None
		try:
			ds = dcmread(dcm_path, stop_before_pixels=True, force=True)
			if hasattr(ds, 'PixelSpacing') and len(ds.PixelSpacing) == 2:
				ps = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
			print(f"[scan:{i}] PixelSpacing={ps}")
		except Exception as e:
			print(f"[scan:{i}] Failed reading PixelSpacing: {e}")

		over_png = os.path.join(args.out, f'sample_overlay_{i:02d}.png')
		try:
			draw_overlay(dcm_path, roi['points'], over_png)
			print(f"[scan:{i}] Wrote overlay: {over_png}")
			item['dicom_found'] = True
			item['overlay_path'] = over_png
			item['pixel_spacing'] = ps
		except Exception as e:
			print(f"[scan:{i}] Overlay failed: {e}")
			missing_pairs.append({'xml_path': xml_path, 'series_uid': series_uid, 'reason': 'overlay_failed'})
		summary['samples'].append(item)

	# Write outputs
	sum_p = os.path.join(args.out, 'summary.json')
	miss_p = os.path.join(args.out, 'missing_pairs.json')
	with open(sum_p, 'w', encoding='utf-8') as f:
		json.dump(summary, f, indent=2)
	with open(miss_p, 'w', encoding='utf-8') as f:
		json.dump(missing_pairs, f, indent=2)

	print(f"[scan] Done. Patients={n_patients}, XML={n_xml}, DICOM(all/CT)={n_dicom_all}/{n_ct}")
	print(f"[scan] Wrote: {sum_p} and {miss_p}")


if __name__ == '__main__':
	main()
