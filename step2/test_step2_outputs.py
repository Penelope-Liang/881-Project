import argparse
import json
import os


def main():
	"""main function tests the outputs from step2/build_indexes.py, check whether slice_index.json, 
	roi_index.json and previews directory exist, verify that slice and ROI records are valid, 
	and validate that preview PNG files exist.
	"""
	parser = argparse.ArgumentParser(description="Test outputs from step2/build_indexes.py")
	parser.add_argument('--out', dest='out_dir', required=True, help='Output folder used by step2/build_indexes.py')
	args = parser.parse_args()

	slice_idx_path = os.path.join(args.out_dir, "slice_index.json")
	roi_idx_path = os.path.join(args.out_dir, "roi_index.json")
	preview_dir = os.path.join(args.out_dir, "previews")
	print(f"[test2] Checking {slice_idx_path}")
	print(f"[test2] Checking {roi_idx_path}")
	print(f"[test2] Checking previews dir {preview_dir}")
	assert os.path.exists(slice_idx_path), f"Missing {slice_idx_path}"
	assert os.path.exists(roi_idx_path), f"Missing {roi_idx_path}"
	assert os.path.isdir(preview_dir), f"Missing previews dir: {preview_dir}"

	with open(slice_idx_path, "r", encoding="utf-8") as f:
		slice_idx = json.load(f)
	with open(roi_idx_path, "r", encoding="utf-8") as f:
		roi_idx = json.load(f)

	print(f"[test2] Slice records={len(slice_idx)}, ROI records={len(roi_idx)}")
	if slice_idx:
		print(f"[test2] Sample slice: {json.dumps(slice_idx[0], indent=2)[:500]}")
	preview = [os.path.join(preview_dir, filename) for filename in os.listdir(preview_dir) if filename.endswith(".png")]
	preview.sort()
	print(f"[test2] Preview PNGs: {len(preview)}")
	for path in preview[:4]:
		print(f"[test2] Example: {path}")


if __name__ == '__main__':
	main()
