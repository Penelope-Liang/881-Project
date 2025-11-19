import argparse
import os
import glob
import json
import numpy as np
from lxml import etree
from pydicom import dcmread
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional, Dict


LIDC_NS = {"lidc": "http://www.nih.gov"}


def ensure_dir(path: str) -> None:
	"""ensure_dir function ensures directory exist

	args:
		path: directory path
	"""
	os.makedirs(path, exist_ok=True)


def parse_lidc_xml(xml_path: str) -> Tuple[Optional[str], List[Dict]]:
	"""parse_lidc_xml function parses LIDC XML file and extract series UID and ROI

	args:
		xml_path: path to LIDC XML file

	returns:
		tuple of series_uid, rois 
		and rois is a list of ROI with sop_uid and points
	"""
	tree = etree.parse(xml_path)
	root = tree.getroot()
	series_uid = None
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


def build_sop_map(series_dir: str) -> Dict[str, str]:
	"""build_sop_map function builds mapping from SOP instance UID to dicom file path

	args:
		series_dir: directory containing dicom files

	returns:
		dict mapping SOP instance UID to file path
	"""
	sop_map: Dict[str, str] = {}
	for path in glob.iglob(os.path.join(series_dir, "**", "*.dcm"), recursive=True):
		try:
			dcm_dataset = dcmread(path, stop_before_pixels=True, force=True)
		except Exception:
			continue
		sop = getattr(dcm_dataset, 'SOPInstanceUID', None)
		if sop:
			sop_map[sop] = path
	return sop_map


def load_and_normalize_dicom(dicom_path: str) -> np.ndarray:
	"""load_and_normalize_dicom function loads dicom file and normalizes img array between 5th and 95th percentiles

	args:
		dicom_path: path to dicom file

	returns:
		normalized img array with values in [0, 1]
	"""
	dcm_dataset = dcmread(dicom_path)
	img_arr = dcm_dataset.pixel_array.astype(np.float32)
	vmin, vmax = np.percentile(img_arr, [5, 95])
	img_arr = np.clip((img_arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
	return img_arr


def draw_overlay(dicom_path: str, points: List[Tuple[int, int]], output_png: str) -> None:
	"""draw_overlay function draws ROI polygon overlay on dicom img and save as PNG

	args:
		dicom_path: path to dicom file
		points: list of (x, y) coordinates for ROI polygon
		output_png: path to output PNG file
	"""
	img_arr = load_and_normalize_dicom(dicom_path)
	img = Image.fromarray((img_arr * 255).astype(np.uint8)).convert("RGB")
	draw = ImageDraw.Draw(img)
	points_int = [(int(x), int(y)) for x, y in points]
	if points_int:
		point2 = points_int + [points_int[0]]
		draw.line(point2, fill=(255, 0, 0), width=2)
	img.save(output_png)


def count_patients(data_root: str) -> int:
	"""count_patients function counts valid number of patient folders with the name starting with LIDC-IDRI-

	args:
		data_root: path to LIDC-IDRI

	returns:
		count: number of valid patient folders
	"""
	count = 0
	for name in os.listdir(data_root):
		p = os.path.join(data_root, name)
		if os.path.isdir(p) and name.startswith("LIDC-IDRI-"):
			count += 1
	return count


def count_dicom_and_ct(data_root: str) -> Tuple[int, int]:
	"""count_dicom_and_ct function counts number of all .dcm files and CT files

	args:
		data_root: path to LIDC-IDRI root

	returns:
		tuple of (all_cnt, ct_cnt) where all_cnt is number of .dcm files found 
		and ct_cnt is number of CT files
	"""
	all_cnt = 0
	ct_cnt = 0
	for root, _, files in os.walk(data_root):
		for name in files:
			if not name.lower().endswith(".dcm"):
				continue
			all_cnt += 1
			path = os.path.join(root, name)
			try:
				dcm_dataset = dcmread(path, stop_before_pixels=True, force=True)
				if getattr(dcm_dataset, "Modality", None) == "CT":
					ct_cnt += 1
			except Exception:
				pass
	return all_cnt, ct_cnt


def main() -> None:
	"""main function scans the LIDC-IDRI dataset catalog, count the number of patients, XML files, and .dcm files, 
	select the first N XML files for checking whether ROI can be found in the corresponding .dcm file, and 
	visualize the ROI overlaid on the DICOM, and save as a PNG, output summary.json and missing_pairs.json.
	"""
	parser = argparse.ArgumentParser(description='Scan LIDC data and validate XML-DICOM pairing with visual proofs')
	parser.add_argument('--data-root', required=True, help='Path to LIDC-IDRI root')
	parser.add_argument('--samples', type=int, default=5, help='Number of XML files to sample for overlay checks')
	parser.add_argument('--out', dest='out_dir', required=True, help='Output directory for summary and overlays')
	args = parser.parse_args()

	print(f"[scan] Starting scan_and_validate with data_root={args.data_root}, samples={args.samples}, out={args.out_dir}")
	ensure_dir(args.out_dir)

	print("[scan] Counting patients, XMLs and dicom files ...")
	n_patients = count_patients(args.data_root)
	xml_files = list(glob.iglob(os.path.join(args.data_root, "**", "*.xml"), recursive=True))
	n_xml = len(xml_files)
	n_dicom_all, n_ct = count_dicom_and_ct(args.data_root)
	print(f"[scan] Found patients={n_patients}, xml={n_xml}, dicom(all/ct)={n_dicom_all}/{n_ct}")

	summary = {
		"num_patients": n_patients,
		"num_xml": n_xml,
		"num_dicom_all": n_dicom_all,
		"num_dicom_ct": n_ct,
		"samples": [],
	}
	missing_pairs = []

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
			"xml_path": xml_path,
			"series_uid": series_uid,
			"roi_count": len(rois) if rois else 0,
			"dicom_found": False,
			"overlay_path": None,
			"pixel_spacing": None,
		}
		if not rois or not series_uid:
			print(f"[scan:{i}] Skip: no_series_or_roi")
			summary["samples"].append(item)
			missing_pairs.append({"xml_path": xml_path, "reason": "no_series_or_roi"})
			continue

		series_dir = os.path.dirname(xml_path)
		sop_map = build_sop_map(series_dir)
		print(f"[scan:{i}] DICOM in series_dir={series_dir}: {len(sop_map)}")
		# pick first ROI that exists in DICOM set
		chosen = None
		for roi in rois:
			p = sop_map.get(roi["sop_uid"])
			if p:
				chosen = (roi, p)
				break
		if not chosen:
			print(f"[scan:{i}] SOP not found for any ROI in this series")
			summary["samples"].append(item)
			missing_pairs.append({"xml_path": xml_path, "series_uid": series_uid, "reason": "sop_not_found"})
			continue

		roi, dicom_path = chosen
		print(f"[scan:{i}] Chosen SOP={roi['sop_uid']} dicom_path={dicom_path}")
		ps = None
		try:
			dcm_dataset = dcmread(dicom_path, stop_before_pixels=True, force=True)
			if hasattr(dcm_dataset, "PixelSpacing") and len(dcm_dataset.PixelSpacing) == 2:
				ps = [float(dcm_dataset.PixelSpacing[0]), float(dcm_dataset.PixelSpacing[1])]
			print(f"[scan:{i}] PixelSpacing={ps}")
		except Exception as e:
			print(f"[scan:{i}] Failed reading PixelSpacing: {e}")

		over_png = os.path.join(args.out_dir, f"sample_overlay_{i:02d}.png")
		try:
			draw_overlay(dicom_path, roi["points"], over_png)
			print(f"[scan:{i}] Wrote overlay: {over_png}")
			item["dicom_found"] = True
			item["overlay_path"] = over_png
			item["pixel_spacing"] = ps
		except Exception as e:
			print(f"[scan:{i}] Overlay failed: {e}")
			missing_pairs.append({"xml_path": xml_path, "series_uid": series_uid, "reason": "overlay_failed"})
		summary["samples"].append(item)

	summary_path = os.path.join(args.out_dir, "summary.json")
	miss_path = os.path.join(args.out_dir, "missing_pairs.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)
	with open(miss_path, "w", encoding="utf-8") as f:
		json.dump(missing_pairs, f, indent=2)

	print(f"[scan] Done. Patients={n_patients}, XML={n_xml}, DICOM(all/CT)={n_dicom_all}/{n_ct}")
	print(f"[scan] Wrote: {summary_path} and {miss_path}")


if __name__ == '__main__':
	main()
