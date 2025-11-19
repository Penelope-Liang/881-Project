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
├── step1/                         # step 1: data scanning and validation
│   ├── scan_and_validate.py     # scan dataset, generate previews
│   ├── test_scan_outputs.py     # test outputs
│   └── README.md
│
├── step2/                         # step 2: index building and rule-based retrieval
│   ├── build_indexes.py         # build slice and ROI indexes
│   ├── compute_diameter_and_update.py  # compute diameter
│   ├── rule_query_export.py     # rule-based query and export   
│   └── test_step2_outputs.py    # test outputs
│
├── step3/                         # step 3: quality control
│   └── build_qc_report.py       # generate QC report
│
├── step4/                         # step 4: UNet segmentation model
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
├── step5/                         # step 5: retrieval model
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
├── step6/                       # step 6: 3D Slicer integration
│   ├── query_to_slicer.py       # csv-based retrieval
│   └── find_and_show.py         # end-to-end retrieval and display
│
├── outputs/                       # all output files
│   ├── scan/                    # step1 outputs
│   ├── step2/                   # step2 indexes and CSVs
│   ├── step3/                   # qc reports
│   ├── step4/                   # unet models and evaluations
│   ├── step5_reg/               # regression models
│   └── step6_*/                 # slicer exports
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
python -m venv .venv
.venv/Scripts/activate  # windows
source .venv/bin/activate  # linux or Mac
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

# 4.2 train UNet model
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
  --model outputs/step4/unet_best.path \
  --out outputs/step4/eval_test \
  --img-size 256
```

**Outputs**:
- `outputs/step4/unet_best.path` - Trained model
- `outputs/step4/eval_test/summary.json` - Evaluation metrics
- `outputs/step4/eval_test/precision_recall_curve.png` - Precision Recall curve

**Evaluation Metrics** (Test set):
- Dice: ~0.85
- IoU: ~0.75
- Precision: ~0.88
- Recall: ~0.84

---

### Step 5: Retrieval Model Training Only for Discussion
#### Option A: Diameter Regression Model (Recommended)
**Purpose**: Train ResNet18 regression model to predict nodule diameter

```bash
# 5A.1 train regression model
python -m step5.regression.train_reg \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --train-patients step4/splits/train_patients.txt \
  --val-patients step4/splits/validation_patients.txt \
  --epochs 20 \
  --bs 32 \
  --img-size 256 \
  --out outputs/step5_reg

# 5A.2 predict on test set
python -m step5.regression.predict_reg \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --patients-file step4/splits/test_patients.txt \
  --model outputs/step5_reg/reg_best.path \
  --out outputs/step5_reg/test_pred \
  --img-size 256
```

**Outputs**:
- `outputs/step5_reg/reg_best.path` - Regression model
- `outputs/step5_reg/test_pred/pred_regression.csv` - Prediction results

**Evaluation Metrics**:
- MAE: ~1.8mm
- Classification accuracy: ~84%

#### Option B: CLIP Contrastive Learning (Alternative)
```bash
# 5B.1 train CLIP
python -m step5.train_clip \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --train-patients step4/splits/train_patients.txt \
  --val-patients step4/splits/validation_patients.txt \
  --epochs 30 \
  --bs 64 \
  --out outputs/step5

# 5B.2 build embedding library
python -m step5.build_embeddings \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --patients-file step4/splits/test_patients.txt \
  --model outputs/step5/clip_best.pth \
  --out outputs/step5/test_embed

# 5B.3 semantic query
python -m step5.semantic_query \
  --emb-dir outputs/step5/test_embed \
  --model outputs/step5/clip_best.pth \
  --query "large nodule" \
  --topk 10 \
  --out outputs/step5/query_result
```

---

### Step 6: 3D Slicer Integration
#### Option A: CSV-based Retrieval
```bash
python -m step6.query_to_slicer \
  --pred-csv outputs/step5_reg/test_pred/pred_regression.csv \
  --query ">=10mm" \
  --topk 10 \
  --unet outputs/step4_test/unet_best.path \
  --out outputs/step6_query
```

#### Option B: End-to-End Retrieval (Recommended)
```bash
python -m step6.find_and_show \
  --data-root LIDC/dataset/LIDC-IDRI/LIDC-IDRI-0001 \
  --query ">=5mm" \
  --topk 10 \
  --unet outputs/step4_test/unet_best.path \
  --out outputs/step6_case \
  --slicer /path/to/Slicer.exe
```

**Outputs**:
- `outputs/step6_*/hit_*.dcm` - Retrieved DICOM slices
- `outputs/step6_*/hit_*_mask.nii.gz` - Segmentation masks in NIFTI format
- `outputs/step6_*/topk_hits.csv` - Retrieval

**3D Slicer Import**:
1. Open 3D Slicer
2. File → Add Data → Select output folder
3. Load DICOM and corresponding mask files
4. Visualize segmentation results in Segment Editor

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
- **Performance**: MAE 1.8mm, accuracy 84%

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

## Documentation
- **Step READMEs**: In each step folder (step1, step5)
- **Quality Reports**: `outputs/step3/REPORT.md` (generated after step3)

---

## Notes

1. **Data Path**: Ensure LIDC-IDRI data is in `LIDC/dataset/LIDC-IDRI/`
2. **Virtual Environment**: Always run in `.venv` to avoid DLL conflicts
3. **GPU Memory**: UNet training requires ~8GB VRAM (batch_size=16)
4. **Patient Splits**: Use patient-level splits to avoid data leakage