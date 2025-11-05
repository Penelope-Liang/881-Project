import argparse
import json
import os


def main():
	parser = argparse.ArgumentParser(description='Test outputs from step2/build_indexes.py')
	parser.add_argument('--out', required=True, help='Output folder used by step2/build_indexes.py')
	args = parser.parse_args()

	slice_p = os.path.join(args.out, 'slice_index.json')
	roi_p = os.path.join(args.out, 'roi_index.json')
	prev_dir = os.path.join(args.out, 'previews')
	print(f"[test2] Checking {slice_p}")
	print(f"[test2] Checking {roi_p}")
	print(f"[test2] Checking previews dir {prev_dir}")
	assert os.path.exists(slice_p), f"Missing {slice_p}"
	assert os.path.exists(roi_p), f"Missing {roi_p}"
	assert os.path.isdir(prev_dir), f"Missing previews dir: {prev_dir}"

	with open(slice_p, 'r', encoding='utf-8') as f:
		slice_idx = json.load(f)
	with open(roi_p, 'r', encoding='utf-8') as f:
		roi_idx = json.load(f)

	print(f"[test2] slice records={len(slice_idx)}, roi records={len(roi_idx)}")
	if slice_idx:
		print(f"[test2] sample slice item: {json.dumps(slice_idx[0], indent=2)[:500]}")
	# list some previews
	prevs = [os.path.join(prev_dir, n) for n in os.listdir(prev_dir) if n.endswith('.png')]
	prevs.sort()
	print(f"[test2] preview pngs: {len(prevs)}")
	for p in prevs[:4]:
		print(f"[test2] example preview: {p}")


if __name__ == '__main__':
	main()
