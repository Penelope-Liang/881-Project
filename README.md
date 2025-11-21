# LIDC-IDRI Pulmonary Nodule Retrieval System
A system designed to structure LIDC-IDRI lung nodule data for keyword or size-based filtering, UNet-based segmentation, and seamless integration with 3D Slicer for visualization.

## Installation
### Get the Code
```bash
git clone <repository-url>
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

## Project Structure
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
│   └── step7_*/                 # slicer visualization outputs
│
├── .venv/                         # python virtual environment
├── requirements.txt               # dependencies list
└── README.md                      # this file
```

## Usage

Environment setup instructions are provided in the following section.

## Environment Setup
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

**Option B: Install with GPU support (CUDA 12.8)**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pydicom lxml Pillow numpy pandas scikit-image matplotlib tqdm SimpleITK
```

### Step 3: Verify Installation
```bash
python -c "import torch; import pydicom; import numpy; print('Installation successful!')"
```

### GPU Support
- Requires CUDA 12.8+ (RTX 5070 sm_120 support)
- PyTorch Nightly version recommended for latest GPU architectures

## Complete Workflow
### Step 1: Data Scanning and Validation
**Purpose**: Validate LIDC-IDRI data integrity and generate sample previews
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

**Outputs**:
- `outputs/scan/summary.json` - Dataset statistics
- `outputs/scan/sample_overlay_*.png` - ROI overlay preview images

---

### Step 2: Index Building and Preprocessing
**Purpose**: Build slice and ROI indexes, compute nodule diameters
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

**Outputs**:
- `outputs/step2/roi_with_diam.csv` - Roi data with diameter
- `outputs/step2/slice_index_with_diam.csv` - Slice data
- `outputs/rule_ge3/topk_hits.csv` - Query results

---

### Step 3: Quality Control
**Purpose**: Generate data quality report
```bash
python -m step3.build_qc_report \
  --in outputs/step2 \
  --out outputs/step3
```

**Outputs**:
- `outputs/step3/diameter_histogram.png` - Diameter distribution histogram
- `outputs/step3/REPORT.md` - QC report

---

### Step 4: UNet Segmentation Model Training
**Purpose**: Train nodule segmentation model

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

**Outputs**:
- `outputs/step4/unet_best.pth` - Trained model
- `outputs/step4/eval_test/summary.json` - Evaluation metrics
- `outputs/step4/eval_test/precision_recall_curve.png` - Precision Recall curve

**Evaluation Metrics** (Test set):
- Dice
- IoU
- Precision
- Recall

---

### Step 5: Retrieval Model Training Only for Discussion
#### Option A: Diameter Regression Model (Recommended)
**Purpose**: Train ResNet18 regression model to predict nodule diameter

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

**Outputs**:
- `outputs/step5_reg/reg_best.pth` - Regression model
- `outputs/step5_reg/test_pred/pred_regression.csv` - Prediction results

**Evaluation Metrics**:
- MAE
- Classification accuracy

#### Option B: CLIP Contrastive Learning (Alternative)
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

### Step 6: YOLOv8 Nodule Detector Training
**Purpose**: Train YOLOv8 model to detect lung nodules on full CT slices

**Prerequisites**:
- Prepared YOLO dataset (images and labels in YOLO format with dataset config YAML)
- Ultralytics YOLOv8 (`pip install ultralytics`)

```bash
# quick test (1 epoch, verified working)
python step6/train_detector.py \
    --data plan_b/detection_data/nodule_detection.yaml \
    --model yolov8s.pt \
    --epochs 1 \
    --batch 2 \
    --device cpu

# full training (50 epochs)
python step6/train_detector.py \
    --data plan_b/detection_data/nodule_detection.yaml \
    --model yolov8s.pt \
    --epochs 50 \
    --batch 16 \
    --device cpu
```

**Arguments**:
- `--data`: Path to YOLO dataset config YAML (required, example: `plan_b/detection_data/nodule_detection.yaml`)
- `--model`: Pretrained YOLOv8 model (default: `yolov8s.pt`)
  - Options: `yolov8n.pt` (nano), `yolov8s.pt` (small), `yolov8m.pt` (medium)
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size (default: 16)
- `--device`: Device to use, `cuda` or `cpu` (default: `cuda`, use `cpu` on Mac or if no GPU available)
- `--project`: Project directory for results (default: `plan_b/runs/detect`)
- `--name`: Experiment name (default: `nodule_yolov8`)

**Outputs**:
- `plan_b/runs/detect/nodule_yolov8/weights/best.pt` - Best model checkpoint
- `plan_b/runs/detect/nodule_yolov8/weights/last.pt` - Last model checkpoint
- `plan_b/runs/detect/nodule_yolov8/results.csv` - Training metrics
- `plan_b/runs/detect/nodule_yolov8/results.png` - Training curves

**Training Time**:
- GPU (RTX 3090): ~2-3 hours for 50 epochs
- CPU: Not recommended (very slow)

---

### Step 7: 3D Slicer Visualization Pipeline
**Purpose**: End-to-end nodule detection, segmentation, and 3D Slicer visualization

**Prerequisites**:
- Trained YOLOv8 detector (from Step 6)
- Trained UNet segmentation model (from Step 4)
- Trained regression model (from Step 5)
- 3D Slicer installed (download from https://www.slicer.org/)

#### Step 7.1: Run Detection and Segmentation Pipeline

```bash
# note: replace placeholders with actual paths
# <path_to_dicom_directory>: directory containing DICOM files
# <path_to_yolo_model>: path to trained YOLOv8 model .pt file
# <path_to_unet_model>: path to trained UNet model .pth file
# <path_to_regression_model>: path to trained regression model .pth file
# <output_directory>: directory to save results

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

**Arguments**:
- `--ct-dir`: Directory containing DICOM files (required)
- `--query`: Query string for filtering nodules (required)
  - Examples: `">3mm"`, `"<5mm"`, `"3-5mm"`
- `--detector`: Path to trained YOLOv8 model `.pt` file (required)
- `--unet`: Path to trained UNet model `.pth` file (required)
- `--regressor`: Path to trained regression model `.pth` file (required)
- `--output-dir`: Output directory for results (default: `plan_b/outputs/result`)
- `--conf-threshold`: YOLOv8 confidence threshold (default: 0.25)
- `--device`: Device to use, `cuda` or `cpu` (default: `cuda`, use `cpu` on Mac or if no GPU available)

**Outputs**:
- `<output_dir>/series.nii.gz` - 3D CT volume
- `<output_dir>/series_label.nii.gz` - 3D segmentation label volume
- `<output_dir>/summary.json` - Detection and segmentation summary

#### Step 7.2: Visualize in 3D Slicer

```bash
python step7/visualize.py \
    --result-dir <output_directory> \
    --slicer-path "<path_to_slicer_executable>"
```

**Arguments**:
- `--result-dir`: Directory containing pipeline results (required)
- `--slicer-path`: Path to 3D Slicer executable (optional, auto-detected if not specified)
  - Auto-detection checks common installation paths for your platform
  - Mac: `/Applications/Slicer.app/Contents/MacOS/Slicer`
  - Windows: `D:/Slicer 5.8.1/Slicer.exe`, `C:/Program Files/Slicer/Slicer.exe`
  - Linux: `/usr/local/Slicer/Slicer`, `/opt/Slicer/Slicer`
- `--output-script`: Custom output path for generated Slicer script (optional)

**What Happens**:
1. Generates a Slicer Python script with visualization commands
2. Automatically launches 3D Slicer with the script
3. Displays CT volume, segmentation, and annotations in four-view layout

**Visualization Features**:
- **2D Views**: CT volume with lung window, red segmentation overlay, yellow centroids, green bounding boxes, cyan contours
- **3D View**: Red 3D segmentation model with all markups visible
- **Layout**: Four-view layout (Axial, Sagittal, Coronal, 3D) with synchronized navigation

**Example Usage**:

```bash
# example 1: complete workflow (pipeline + visualization)
# step 1: run detection and segmentation pipeline
python step7/pipeline.py \
    --ct-dir outputs/ct_case \
    --query ">5mm" \
    --detector plan_b/runs/detect/nodule_yolov813/weights/best.pt \
    --unet outputs/step4/unet_best.pth \
    --regressor outputs/step5_reg/reg_best.path \
    --output-dir plan_b/outputs/large_nodules \
    --device cpu

# step 2: visualize the results
# slicer path is auto-detected, or specify manually with --slicer-path
python step7/visualize.py \
    --result-dir plan_b/outputs/large_nodules

# example 2: quick test visualization only (skip pipeline)
# uses pre-generated example_output, slicer path auto-detected
python step7/visualize.py \
    --result-dir step7/example_output
```

**Performance**:
- Pipeline Runtime (GPU): ~30-60 seconds per CT volume
- Pipeline Runtime (CPU): ~5-10 minutes per CT volume
- Slicer Launch: ~10-20 seconds

---

## Core Models
### 1. UNet Segmentation Model (Step 4)
- **Architecture**: 2D UNet (4-layer encoder-decoder)
- **Input**: 256×256 single-channel CT slice
- **Output**: 256×256 binary segmentation mask
- **Training**: BCE + Dice Loss, Adam optimizer
- **Performance**: Dice 0.85, IoU 0.75

### 2. Diameter Regression Model (Step 5A) for Discussion
- **Architecture**: ResNet18 + dual-head (regression + classification)
- **Input**: 256×256 single-channel ROI image
- **Output**: Nodule diameter and diameter class
- **Training**: MSE (regression) + CrossEntropy (classification), Adam optimizer

### 3. CLIP Contrastive Learning Model (Step 5B, Alternative) for Discussion
- **Architecture**: ResNet18 (image) + LSTM (text)
- **Training**: InfoNCE contrastive loss
- **Performance**: Weak, not recommended for production

---

## Training Procedures
### Model Training Overview

This project includes training procedures for two main models:

1. **UNet Segmentation Model** (Step 4) - For nodule segmentation
2. **Diameter Regression Model** (Step 5A) - For diameter prediction

See detailed training commands in the [Complete Workflow](#complete-workflow) section above.

### Training Tips
- Use patient-level splits to avoid data leakage
- Monitor GPU memory usage (UNet requires ~8GB VRAM)
- Save checkpoints regularly during training
- Use validation set to prevent overfitting

---

## Notes

1. **Data Path**: Ensure LIDC-IDRI data is in `LIDC/dataset/LIDC-IDRI/`
2. **Virtual Environment**: Always run in `.venv` to avoid DLL conflicts
3. **GPU Memory**: UNet training requires ~8GB VRAM (batch_size=16)
4. **Patient Splits**: Use patient-level splits to avoid data leakage