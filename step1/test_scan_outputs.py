import argparse
import json
import os


def main():
	parser = argparse.ArgumentParser(description='Test outputs from scan_and_validate.py')
	parser.add_argument('--out', required=True, help='Output directory used by scan_and_validate.py')
	args = parser.parse_args()

	summary_p = os.path.join(args.out, 'summary.json')
	missing_p = os.path.join(args.out, 'missing_pairs.json')
	print(f"[test] Checking existence: {summary_p}")
	print(f"[test] Checking existence: {missing_p}")
	assert os.path.exists(summary_p), f"Missing {summary_p}"
	assert os.path.exists(missing_p), f"Missing {missing_p}"

	with open(summary_p, 'r', encoding='utf-8') as f:
		summary = json.load(f)
	print(f"[test] Stats: patients={summary.get('num_patients')}, xml={summary.get('num_xml')}, dicom(all/ct)={summary.get('num_dicom_all')}/{summary.get('num_dicom_ct')}")
	assert 'num_patients' in summary and summary['num_patients'] >= 0
	assert 'num_xml' in summary and summary['num_xml'] >= 0
	assert 'num_dicom_all' in summary and summary['num_dicom_all'] >= 0
	assert 'num_dicom_ct' in summary and summary['num_dicom_ct'] >= 0
	assert isinstance(summary.get('samples', []), list)

	ok_overlays = 0
	first_overlay = None
	for s in summary.get('samples', []):
		if s.get('dicom_found') and s.get('overlay_path'):
			assert os.path.exists(s['overlay_path']), f"Overlay file not found: {s['overlay_path']}"
			ok_overlays += 1
			if first_overlay is None:
				first_overlay = s['overlay_path']

	print(f"[test] Overlays found: {ok_overlays}/{len(summary.get('samples', []))}")
	if first_overlay:
		print(f"[test] Example overlay: {first_overlay}")


if __name__ == '__main__':
	main()
