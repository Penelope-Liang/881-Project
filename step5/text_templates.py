import random

# paraphrases per size bin
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


def bin_label_for_diameter(diameter_mm: float) -> str:
	"""bin_label_for_diameter function returns size bin label for given diameter

	args:
		diameter_mm: nodule diameter in millimeters

	returns:
		size bin label string ("<=6", "6-10", or ">10")
	"""
	if diameter_mm <= 6.0:
		return "<=6"
	elif diameter_mm <= 10.0:
		return "6-10"
	return ">10"


def size_text_for_diameter(diameter_mm: float) -> str:
	"""size_text_for_diameter function returns deterministic primary text template for given diameter

	args:
		diameter_mm: nodule diameter in millimeters

	returns:
		primary text description string for the diameter's size bin
	"""
	bin = bin_label_for_diameter(diameter_mm)
	return BIN_PARAPHRASES[bin][0]


def sample_text_for_diameter(diameter_mm: float, rand: bool = True) -> str:
	"""sample_text_for_diameter function returns randomly sampled or deterministic text template for given diameter

	args:
		diameter_mm: nodule diameter in millimeters
		rand: whether to randomly sample from available templates (default: True)

	returns:
		text description string for the diameter's size bin (random or first)
	"""
	bin = bin_label_for_diameter(diameter_mm)
	cands = BIN_PARAPHRASES[bin]
	return random.choice(cands) if rand else cands[0]


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
