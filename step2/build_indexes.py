import argparse
import os
import glob
import json
from typing import List, Tuple, Dict, Optional
import numpy as np
from lxml import etree
from pydicom import dcmread
from PIL import Image
from skimage.draw import polygon as sk_polygon

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
	"""build_sop_map function builds mapping from SOP instance UID to dcm file path

	args:
		series_dir: directory containing dcm files

	returns:
		dict mapping SOP instance UID to file path
	"""
	sop_map: Dict[str, str] = {}
	for path in glob.iglob(os.path.join(series_dir, "**", "*.dcm"), recursive=True):
		try:
			dcm_dataset = dcmread(path, stop_before_pixels=True, force=True)
			sop = getattr(dcm_dataset, "SOPInstanceUID", None)
			if sop:
				sop_map[sop] = path
		except Exception:
			continue
	return sop_map


def make_slice_mask(shape: Tuple[int, int], roi_list: List[List[Tuple[int, int]]]) -> np.ndarray:
	"""make_slice_mask function creates binary mask from ROI

	args:
		shape: img shape (height, width)
		roi_list: list of ROI polygons, (x, y)

	returns:
		binary mask array with ROI set to 255
	"""
	mask = np.zeros(shape, dtype=np.uint8)
	for points in roi_list:
		r = np.array([point[1] for point in points])
		c = np.array([point[0] for point in points])
		rr, cc = sk_polygon(r, c, shape=shape)
		mask[rr, cc] = 255
	return mask


def write_png(arr: np.ndarray, path: str) -> None:
	"""write_png function saves numpy array as PNG img

	args:
		arr: img array
		path: output file path
	"""
	Image.fromarray(arr).save(path)


def main():
	"""main function builds slice and ROI indexes with masks and previews from LIDC-IDRI dataset
	"""
	parser = argparse.ArgumentParser(description='Step2: Build slice-level and ROI-level indexes with masks and previews')
	parser.add_argument('--data-root', required=True)
	parser.add_argument('--out', dest='out_dir', required=True, help='Output folder for indexes and previews')
	parser.add_argument('--preview', type=int, default=10, help='Max number of slice previews to write')
	args = parser.parse_args()

	print(f"[step2] building indexes from {args.data_root}")
	ensure_dir(args.out_dir)
	ensure_dir(os.path.join(args.out_dir, "previews"))

	xml_files = list(glob.iglob(os.path.join(args.data_root, "**", "*.xml"), recursive=True))
	print(f"[step2] found xml files: {len(xml_files)}")

	slice_idx: List[Dict] = []
	roi_idx: List[Dict] = []
	preview_written = 0

	for xml_path in xml_files:
		series_uid, rois = None, []
		try:
			series_uid, rois = parse_lidc_xml(xml_path)
		except Exception as e:
			print(f"[step2] parse_lidc_xml failed: {xml_path} err={e}")
			continue
		if not series_uid or not rois:
			continue
		series_dir = os.path.dirname(xml_path)
		sop_map = build_sop_map(series_dir)
		sop_to_points: Dict[str, List[List[Tuple[int, int]]]] = {}
		for r in rois:
			path = sop_map.get(r["sop_uid"])
			if not path:
				continue
			sop_to_points.setdefault(r["sop_uid"], []).append(r["points"])
		for sop, all_points in sop_to_points.items():
			dicom_path = sop_map[sop]
			try:
				dcm_dataset = dcmread(dicom_path)
			except Exception as e:
				print(f"[step2] read dicom failed: {dicom_path} err={e}")
				continue
			# img = dcm_dataset.pixel_array.astype(np.uint8)
			shape = (int(dcm_dataset.Rows), int(dcm_dataset.Columns))
			mask = make_slice_mask(shape, all_points)
			slice_item = {
				"xml_path": xml_path,
				"series_uid": series_uid,
				"sop_uid": sop,
				"dicom_path": dicom_path,
				"pixel_spacing": [float(dcm_dataset.PixelSpacing[0]), float(dcm_dataset.PixelSpacing[1])] if hasattr(dcm_dataset, "PixelSpacing") and len(dcm_dataset.PixelSpacing) == 2 else None,
				"num_rois": len(all_points),
			}
			slice_idx.append(slice_item)
			for points in all_points:
				roi_idx.append({
					"xml_path": xml_path,
					"series_uid": series_uid,
					"sop_uid": sop,
					"dicom_path": dicom_path,
					"points": points,
				})
			if preview_written < args.preview:
				img_norm = dcm_dataset.pixel_array.astype(np.float32)
				vmin, vmax = np.percentile(img_norm, [5, 95])
				img_norm = np.clip((img_norm - vmin) / (vmax - vmin + 1e-6) * 255.0, 0, 255).astype(np.uint8)
				write_png(img_norm, os.path.join(args.out_dir, "previews", f"slice_{preview_written:03d}_img.png"))
				write_png(mask, os.path.join(args.out_dir, "previews", f"slice_{preview_written:03d}_mask.png"))
				preview_written += 1

	with open(os.path.join(args.out_dir, "slice_index.json"), "w", encoding="utf-8") as f:
		json.dump(slice_idx, f, indent=2)
	with open(os.path.join(args.out_dir, "roi_index.json"), "w", encoding="utf-8") as f:
		json.dump(roi_idx, f, indent=2)

	print(f"[step2] slice_index records={len(slice_idx)}, roi_index records={len(roi_idx)}")
	print(f"[step2] previews written={preview_written}, out={os.path.join(args.out_dir, 'previews')}")


if __name__ == '__main__':
	main()
