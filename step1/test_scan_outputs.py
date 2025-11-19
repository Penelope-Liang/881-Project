import argparse
import json
import os


def main():
	"""test the outputs from scan_and_validate.py, check whether summary.json and missing_pairs.json exist,
	the number of patients, XML files, and .dcm all or CT files are non-negative,
	check samples is a list, and validate that overlay PNG files exist.
	"""
	parser = argparse.ArgumentParser(description='Test outputs from scan_and_validate.py')
	parser.add_argument('--out', dest='out_dir', required=True, help='Output directory used by scan_and_validate.py')
	args = parser.parse_args()

	summary_path = os.path.join(args.out_dir, "summary.json")
	miss_path = os.path.join(args.out_dir, "missing_pairs.json")
	print(f"[test] Checking exist: {summary_path}")
	print(f"[test] Checking exist: {miss_path}")
	assert os.path.exists(summary_path), f"Missing {summary_path}"
	assert os.path.exists(miss_path), f"Missing {miss_path}"

	with open(summary_path, "r", encoding="utf-8") as f:
		summary = json.load(f)
	print(f"[test] Stats: patients={summary.get('num_patients')}, xml={summary.get('num_xml')}, dicom(all/ct)={summary.get('num_dicom_all')}/{summary.get('num_dicom_ct')}")
	assert "num_patients" in summary and summary["num_patients"] >= 0
	assert "num_xml" in summary and summary["num_xml"] >= 0
	assert "num_dicom_all" in summary and summary["num_dicom_all"] >= 0
	assert "num_dicom_ct" in summary and summary["num_dicom_ct"] >= 0
	assert isinstance(summary.get("samples", []), list)

	valid_overlays = 0
	first_overlay = None
	for s in summary.get("samples", []):
		if s.get("dicom_found") and s.get("overlay_path"):
			assert os.path.exists(s["overlay_path"]), f"Overlay file not found: {s['overlay_path']}"
			valid_overlays += 1
			if first_overlay is None:
				first_overlay = s["overlay_path"]

	print(f"[test] Overlays found: {valid_overlays}/{len(summary.get('samples', []))}")
	if first_overlay:
		print(f"[test] Example overlay: {first_overlay}")


if __name__ == '__main__':
	main()
