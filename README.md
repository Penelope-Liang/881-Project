# Overview
LIDC-IDRI Nodule Retrieval System is an end-to-end pipeline to structure LIDC-IDRI lung nodule data for keyword or size-based filtering, UNet-based segmentation, and diameter regression and semantic retrieval, and use 3D Slicer for visualization.

## Installation
```bash
git clone https://github.com/schorm/881_project.git
cd 881_project
```

### Version Information
- **Python**: 3.8+
- **PyTorch**: 2.5.1
- **CUDA**: 12.8+ (optional, for GPU support)

### Prerequisites
- Python 3.8 or higher
- pip package
- 3D Slicer for visualization
- (Optional) CUDA 12.8+ compatible GPU for training

## Code Structure
```
881_project/
├── LIDC/                          # data directory
│   └── dataset/
│       ├── LIDC-IDRI/            # original DICOM data and XML annotations, for data sample
│       └── metadata.csv
│
├── step1/                       # step 1: data scanning and validation
│   ├── scan_and_validate.py     # scan dataset, generate previews
│   ├── test_scan_outputs.py     # test outputs
│   └── README.md
│
├── step2/                       # step 2: index building and rule-based retrieval
│   ├── build_indexes.py         # build slice and ROI indexes
│   ├── compute_diameter_and_update.py  # compute diameter
│   ├── rule_query_export.py     # rule-based query and export   
│   └── test_step2_outputs.py    # test outputs
│
├── step3/                       # step 3: quality control
│   └── build_qc_report.py       # generate QC report
│
├── step4/                       # step 4: UNet segmentation model
│   ├── models/
│   │   └── unet.py              # unet architecture
│   ├── dataset_roi.py           # roi dataset
│   ├── make_patient_splits.py   # patient-level splits
│   ├── train_unet.py            # training script
│   ├── eval_unet.py             # evaluation script
│   ├── infer_unet.py            # inference script
│   ├── Quality_Eval.ipynb       # quality evaluation notebook
│   ├── splits/                  # data splits
│   │   ├── train_patients.txt
│   │   ├── validation_patients.txt
│   │   └── test_patients.txt
│   └── README.md
│
├── step5/                        # step 5: retrieval model
│   ├── regression/               # 5A: diameter regression model (recommended)
│   │   ├── dataset_reg.py
│   │   ├── models_reg.py
│   │   ├── train_reg.py
│   │   └── predict_reg.py
│   ├── models.py                # 5B: CLIP contrastive learning (alternative)
│   ├── dataset.py
│   ├── train_clip.py
│   ├── build_embeddings.py
│   ├── semantic_query.py
│   ├── text_templates.py
│   └── README.md
│
├── step6/                       # step 6: YOLOv8 nodule detector training
│   ├── train_detector.py        # train yolov8 detector
│   └── README.md
│
├── step7/                       # step 7: 3D Slicer visualization pipeline
│   ├── pipeline.py              # end-to-end detection and segmentation
│   ├── visualize.py             # slicer visualization script generator
│   ├── test_visualization.py    # test script
│   ├── example_output/          # example output files
│   └── README.md
│
├── outputs/                     # all output files
│   ├── scan/                    # step1 outputs
│   ├── step2/                   # step2 indexes and CSVs
│   ├── step3/                   # qc reports
│   ├── step4/                   # unet models and evaluations
│   ├── step5_reg/               # regression models
│   ├── step6_*/                 # step6 related outputs
│   └── step7_*/                 # slicer visualization outputs
│
├── plan_b/                      # step6 YOLOv8 detector outputs
│   └── runs/detect/             # YOLOv8 training results and models
│
├── .venv/                         # python virtual environment
├── requirements.txt               # dependencies list
└── README.md                      # this file
```

## Usage
**Note**: On macOS and Linux, you may need to use `python3` instead of `python`. If `python` command is not found, replace all `python` commands with `python3` in the following instructions.


## Quick Setup
### Step 1: Create Virtual Environment
```bash
python3 -m venv .venv
.venv/Scripts/activate  # windows
source .venv/bin/activate  # linux or mac
```

### Step 2: Install Dependencies
**Option A: Install from requirements.txt (CPU-only)**
```bash
pip install -r requirements.txt
```

**Option B: Install with GPU support (CUDA 12.8) or directly use CPU**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pydicom lxml Pillow numpy pandas scikit-image matplotlib tqdm SimpleITK
```

### Step 3: Verify Installation
```bash
python -c "import torch; import pydicom; import numpy; print('Installation successful!')"
```

### GPU Support
- Requires CUDA 12.8+
- PyTorch Nightly version recommended for latest GPU architectures

## Pipeline
### Step 1: Dataset and Annotation Alignment
```bash
# activate virtual environment
.venv/Scripts/activate

# run scan
python -m step1.scan_and_validate \
  --data-root LIDC/dataset/LIDC-IDRI \
  --samples 5 \
  --out outputs/scan

# test outputs
python -m step1.test_scan_outputs --out outputs/scan
```

---

### Step 2: Nodule Index Construction and Geometric Feature Extraction
```bash
# 2.1 build indexes
python -m step2.build_indexes \
  --data-root LIDC/dataset/LIDC-IDRI \
  --out outputs/step2

# 2.2 compute diameter
python -m step2.compute_diameter_and_update \
  --in outputs/step2 \
  --out outputs/step2

# 2.3 query example
python -m step2.rule_query_export \
  --index outputs/step2/slice_index_with_diam.csv \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --query ">=3mm" \
  --topk 10 \
  --out outputs/rule_ge3
```

---

### Step 3: Quality Control
```bash
python -m step3.build_qc_report \
  --in outputs/step2 \
  --out outputs/step3
```

---

### Step 4: Tumour Segmentation Model
```bash
# 4.1 generate patient splits
python -m step4.make_patient_splits \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --out step4/splits

# 4.2 train unet model
python -m step4.train_unet \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --train-patients step4/splits/train_patients.txt \
  --val-patients step4/splits/validation_patients.txt \
  --epochs 30 \
  --bs 16 \
  --img-size 256 \
  --out outputs/step4

# 4.3 evaluate model
python -m step4.eval_unet \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --patients-file step4/splits/test_patients.txt \
  --model outputs/step4/unet_best.pth \
  --out outputs/step4/eval_test \
  --img-size 256
```

---

### Step 5: Slice Retrieval
#### Option A: Diameter Regression Model (Recommended)
```bash
# 5a.1 train regression model
python -m step5.regression.train_reg \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --train-patients step4/splits/train_patients.txt \
  --val-patients step4/splits/validation_patients.txt \
  --epochs 20 \
  --bs 32 \
  --img-size 256 \
  --out outputs/step5_reg

# 5a.2 predict on test set
python -m step5.regression.predict_reg \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --patients-file step4/splits/test_patients.txt \
  --model outputs/step5_reg/reg_best.pth \
  --out outputs/step5_reg/test_pred \
  --img-size 256
```

#### Option B: CLIP Contrastive Learning (Alternative for Discussion)
```bash
# 5b.1 train clip
python -m step5.train_clip \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --train-patients step4/splits/train_patients.txt \
  --val-patients step4/splits/validation_patients.txt \
  --epochs 30 \
  --bs 64 \
  --out outputs/step5

# 5b.2 build embedding library
python -m step5.build_embeddings \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --patients-file step4/splits/test_patients.txt \
  --model outputs/step5/clip_best.pth \
  --out outputs/step5/test_embed

# 5b.3 semantic query
python -m step5.semantic_query \
  --emb-dir outputs/step5/test_embed \
  --model outputs/step5/clip_best.pth \
  --query "large nodule" \
  --topk 5 \
  --out outputs/step5/query_result_test
```

---

### Step 6: YOLOv8 Nodule Detector
**Prerequisites**:
- Prepared YOLO dataset
- Ultralytics YOLOv8 (`pip install ultralytics`)

```bash
python step6/train_detector.py \
    --data plan_b/detection_data/nodule_detection.yaml \
    --model yolov8s.pt \
    --epochs 50 \
    --batch 16 \
    --device cpu
```

---

### Step 7: 3D Slicer Integration
**Prerequisites**:
- Trained YOLOv8 detector (from Step 6)
- Trained UNet segmentation model (from Step 4)
- Trained regression model (from Step 5)
- 3D Slicer installed (download from https://www.slicer.org/)

#### Step 7.1: Run Detection and Segmentation Pipeline
```bash
python step7/pipeline.py \
    --ct-dir <path_to_dicom_directory> \
    --query "<query_string>" \
    --detector <path_to_yolo_model> \
    --unet <path_to_unet_model> \
    --regressor <path_to_regression_model> \
    --output-dir <output_directory> \
    --conf-threshold 0.25 \
    --device cpu
```
# example:
```bash
python step7/pipeline.py \
    --ct-dir outputs/ct_case \
    --query ">5mm" \
    --detector plan_b/runs/detect/nodule_yolov813/weights/best.pt \
    --unet outputs/step4/unet_best.pth \
    --regressor outputs/step5_reg/reg_best.path \
    --output-dir plan_b/outputs/large_nodules \
    --device cpu
```

#### Step 7.2: Visualize in 3D Slicer

```bash
# note: on macOS/Linux, use python3 instead of python if python command is not found
python step7/visualize.py \
    --result-dir <output_directory> \
    --slicer-path "<path_to_slicer_executable>"
```
