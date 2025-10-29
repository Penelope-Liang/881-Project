import random

# Paraphrases per size bin
BIN_PARAPHRASES = {
	"<=6": [
		"a lung nodule with diameter <= 6 mm",
		"a small lung nodule (<= 6 mm)",
		"a pulmonary nodule of small size (<= 6 mm)",
	],
	"6-10": [
		"a lung nodule with diameter 6-10 mm",
		"a medium-size lung nodule (6-10 mm)",
		"a pulmonary nodule of moderate size (6-10 mm)",
	],
	">10": [
		"a lung nodule with diameter > 10 mm",
		"a large lung nodule (> 10 mm)",
		"a pulmonary nodule of large size (> 10 mm)",
	],
}

BIN_RULES = [
	("<=6", -1, 6.0),
	("6-10", 6.0, 10.0),
	(">10", 10.0, 1e9),
]


def bin_label_for_diameter(d_mm: float) -> str:
	if d_mm <= 6.0:
		return "<=6"
	elif d_mm <= 10.0:
		return "6-10"
	return ">10"


def size_text_for_diameter(d_mm: float) -> str:
	"""Deterministic primary template (first of bin)."""
	b = bin_label_for_diameter(d_mm)
	return BIN_PARAPHRASES[b][0]


def sample_text_for_diameter(d_mm: float, rnd: bool = True) -> str:
	b = bin_label_for_diameter(d_mm)
	cands = BIN_PARAPHRASES[b]
	return random.choice(cands) if rnd else cands[0]


def all_query_list() -> list:
	return [
		">=3 mm",
		"<=6 mm",
		"6-10 mm",
		">10 mm",
		# canonical paraphrases
		*BIN_PARAPHRASES["<=6"],
		*BIN_PARAPHRASES["6-10"],
		*BIN_PARAPHRASES[">10"],
	]
