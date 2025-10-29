<<<<<<< HEAD
# 881_project
=======
# LIDC-IDRI 肺结节检索系统

基于LIDC-IDRI数据集的肺结节关键词检索与3D Slicer可视化系统

## 📁 项目结构

```
S:\881project\
├── LIDC/                          # 数据目录
│   └── dataset/
│       ├── LIDC-IDRI/            # 原始DICOM数据和XML标注
│       └── metadata.csv
│
├── step1/                         # 步骤1: 数据扫描与验证
│   ├── scan_and_validate.py     # 扫描数据集,生成预览
│   ├── test_scan_outputs.py     # 测试输出
│   └── README.md
│
├── step2/                         # 步骤2: 索引构建与规则检索
│   ├── build_indexes.py         # 构建slice和ROI索引
│   ├── compute_diameter_and_update.py  # 计算直径
│   ├── rule_query_export.py     # 规则检索导出
│   ├── test_step2_outputs.py   # 测试输出
│   └── README.md
│
├── step3/                         # 步骤3: 质量控制
│   └── build_qc_report.py       # 生成QC报告
│
├── step4/                         # 步骤4: UNet分割模型
│   ├── models/
│   │   └── unet.py              # UNet架构
│   ├── dataset_roi.py           # ROI数据集
│   ├── make_patient_splits.py   # 患者级别划分
│   ├── train_unet.py            # 训练脚本
│   ├── eval_unet.py             # 评估脚本
│   ├── infer_unet.py            # 推理脚本
│   ├── Quality_Eval.ipynb       # 质量评估notebook
│   ├── splits/                  # 数据划分
│   │   ├── train_patients.txt
│   │   ├── val_patients.txt
│   │   └── test_patients.txt
│   └── README.md
│
├── step5/                         # 步骤5: 检索模型
│   ├── regression/               # 5A: 直径回归模型(推荐)
│   │   ├── dataset_reg.py
│   │   ├── models_reg.py
│   │   ├── train_reg.py
│   │   └── predict_reg.py
│   ├── models.py                # 5B: CLIP对比学习(备选)
│   ├── dataset.py
│   ├── train_clip.py
│   ├── build_embeddings.py
│   ├── semantic_query.py
│   ├── text_templates.py
│   └── README.md
│
├── step6/                         # 步骤6: 3D Slicer集成
│   ├── query_to_slicer.py       # 基于预测CSV的检索
│   └── find_and_show.py         # 端到端检索与显示
│
├── outputs/                       # 所有输出文件
│   ├── scan/                    # Step1输出
│   ├── step2/                   # Step2索引和CSV
│   ├── step3/                   # QC报告
│   ├── step4/                   # UNet模型和评估
│   ├── step5_reg/               # 回归模型
│   └── step6_*/                 # Slicer导出
│
├── docs/                          # 文档
│   ├── REPORT_总览与复现实操.md  # 完整项目报告
│   └── PROJECT_SUMMARY.md        # 项目摘要(英文)
│
├── .venv/                         # Python虚拟环境
├── requirements.txt               # 依赖列表
└── README.md                      # 本文件
```

## 🔄 完整工作流程

### Step 1: 数据扫描与验证
**目的**: 验证LIDC-IDRI数据完整性,生成样本预览

```bash
# 激活虚拟环境
.venv/Scripts/activate

# 运行扫描
python -m step1.scan_and_validate \
  --data-root LIDC/dataset/LIDC-IDRI \
  --samples 5 \
  --out outputs/scan

# 测试输出
python -m step1.test_scan_outputs --out outputs/scan
```

**输出**:
- `outputs/scan/summary.json` - 数据集统计
- `outputs/scan/sample_overlay_*.png` - ROI叠加预览图

---

### Step 2: 索引构建与预处理
**目的**: 构建切片和ROI索引,计算结节直径

```bash
# 2.1 构建索引
python -m step2.build_indexes \
  --data-root LIDC/dataset/LIDC-IDRI \
  --out outputs/step2

# 2.2 计算直径
python -m step2.compute_diameter_and_update \
  --roi-json outputs/step2/roi_index.json \
  --slice-json outputs/step2/slice_index.json \
  --out outputs/step2

# 2.3 规则检索示例
python -m step2.rule_query_export \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --query ">=3" \
  --topk 10 \
  --out outputs/rule_ge3
```

**输出**:
- `outputs/step2/roi_with_diam.csv` - ROI级别数据(含直径)
- `outputs/step2/slice_index_with_diam.csv` - 切片级别数据
- `outputs/rule_ge3/topk_hits.csv` - 检索结果

---

### Step 3: 质量控制
**目的**: 生成数据质量报告

```bash
python -m step3.build_qc_report \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --out outputs/step3
```

**输出**:
- `outputs/step3/diameter_hist.png` - 直径分布直方图
- `outputs/step3/REPORT.md` - QC报告

---

### Step 4: UNet分割模型训练
**目的**: 训练结节分割模型

```bash
# 4.1 生成患者划分
python -m step4.make_patient_splits \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --out step4/splits

# 4.2 训练UNet
python -m step4.train_unet \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --train-patients step4/splits/train_patients.txt \
  --val-patients step4/splits/val_patients.txt \
  --epochs 30 \
  --bs 16 \
  --img-size 256 \
  --out outputs/step4

# 4.3 评估模型
python -m step4.eval_unet \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --patients-file step4/splits/test_patients.txt \
  --model outputs/step4/unet_best.pth \
  --out outputs/step4/eval_test \
  --img-size 256
```

**输出**:
- `outputs/step4/unet_best.pth` - 训练好的模型
- `outputs/step4/eval_test/summary.json` - 评估指标(Dice, IoU等)
- `outputs/step4/eval_test/pr_curve.png` - PR曲线

**评估指标** (测试集):
- Dice: ~0.85
- IoU: ~0.75
- Precision: ~0.88
- Recall: ~0.84

---

### Step 5: 检索模型训练

#### 方案A: 直径回归模型 (推荐)
**目的**: 训练ResNet18回归模型预测结节直径

```bash
# 5A.1 训练回归模型
python -m step5.regression.train_reg \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --train-patients step4/splits/train_patients.txt \
  --val-patients step4/splits/val_patients.txt \
  --epochs 20 \
  --bs 32 \
  --img-size 256 \
  --out outputs/step5_reg

# 5A.2 预测测试集
python -m step5.regression.predict_reg \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --patients-file step4/splits/test_patients.txt \
  --model outputs/step5_reg/reg_best.pth \
  --out outputs/step5_reg/test_pred \
  --img-size 256
```

**输出**:
- `outputs/step5_reg/reg_best.pth` - 回归模型
- `outputs/step5_reg/test_pred/pred_regression.csv` - 预测结果

**评估指标**:
- MAE: ~1.8mm
- 分类准确率: ~84%

#### 方案B: CLIP对比学习 (备选)
```bash
# 5B.1 训练CLIP
python -m step5.train_clip \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --train-patients step4/splits/train_patients.txt \
  --val-patients step4/splits/val_patients.txt \
  --epochs 30 \
  --bs 64 \
  --out outputs/step5

# 5B.2 构建嵌入库
python -m step5.build_embeddings \
  --roi-csv outputs/step2/roi_with_diam.csv \
  --patients-file step4/splits/test_patients.txt \
  --model outputs/step5/clip_best.pth \
  --out outputs/step5/test_embed

# 5B.3 语义检索
python -m step5.semantic_query \
  --embed-dir outputs/step5/test_embed \
  --model outputs/step5/clip_best.pth \
  --query "large nodule" \
  --topk 10
```

---

### Step 6: 3D Slicer集成

#### 方案A: 基于预测CSV检索
```bash
python -m step6.query_to_slicer \
  --pred-csv outputs/step5_reg/test_pred/pred_regression.csv \
  --query ">=10" \
  --topk 10 \
  --unet-model outputs/step4/unet_best.pth \
  --out outputs/step6_query
```

#### 方案B: 端到端检索(推荐)
```bash
python -m step6.find_and_show \
  --dicom-root LIDC/dataset/LIDC-IDRI/LIDC-IDRI-0001 \
  --query ">=5" \
  --topk 10 \
  --unet-model outputs/step4/unet_best.pth \
  --reg-model outputs/step5_reg/reg_best.pth \
  --out outputs/step6_case \
  --launch-slicer
```

**输出**:
- `outputs/step6_*/hit_*.dcm` - 检索到的DICOM切片
- `outputs/step6_*/hit_*_mask.nii.gz` - 分割掩码(NIfTI格式)
- `outputs/step6_*/topk_hits.csv` - 检索清单

**3D Slicer导入**:
1. 打开3D Slicer
2. File → Add Data → 选择输出文件夹
3. 加载DICOM和对应的mask文件
4. 在Segment Editor中可视化分割结果

---

## 🔧 环境配置

### 创建虚拟环境
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
```

### 安装依赖
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pydicom lxml Pillow numpy pandas scikit-image matplotlib tqdm SimpleITK
```

### GPU支持
- 需要CUDA 12.8+ (RTX 5070 sm_120支持)
- PyTorch Nightly版本

---

## 📊 核心AI模型

### 1. UNet分割模型 (Step 4)
- **架构**: 2D UNet (4层编码器-解码器)
- **输入**: 256×256单通道CT切片
- **输出**: 256×256二值分割掩码
- **训练**: BCE + Dice Loss, Adam优化器
- **性能**: Dice 0.85, IoU 0.75

### 2. 直径回归模型 (Step 5A)
- **架构**: ResNet18 + 双头(回归+分类)
- **输入**: 256×256单通道ROI图像
- **输出**: 直径(mm) + 直径bin(0-3mm, 3-10mm, 10-20mm, >20mm)
- **训练**: MSE(回归) + CrossEntropy(分类), Adam优化器
- **性能**: MAE 1.8mm, 准确率 84%

### 3. CLIP对比学习模型 (Step 5B, 备选)
- **架构**: ResNet18(图像) + LSTM(文本)
- **训练**: InfoNCE对比损失
- **性能**: 较弱,不推荐用于生产

---

## 📖 详细文档

- **完整报告**: `docs/REPORT_总览与复现实操.md` (中文,含所有细节)
- **项目摘要**: `docs/PROJECT_SUMMARY.md` (英文)
- **各步骤README**: 每个step文件夹内

---

## 🎯 快速开始

```bash
# 1. 激活环境
.venv/Scripts/activate

# 2. 运行完整流程(假设已有训练好的模型)
python -m step6.find_and_show \
  --dicom-root LIDC/dataset/LIDC-IDRI/LIDC-IDRI-0001 \
  --query ">=5" \
  --topk 10 \
  --unet-model outputs/step4/unet_best.pth \
  --reg-model outputs/step5_reg/reg_best.pth \
  --out outputs/demo \
  --launch-slicer
```

---

## 📝 注意事项

1. **数据路径**: 确保LIDC-IDRI数据在 `LIDC/dataset/LIDC-IDRI/`
2. **虚拟环境**: 始终在 `.venv` 中运行,避免DLL冲突
3. **GPU内存**: UNet训练需要~8GB显存(batch_size=16)
4. **患者划分**: 使用患者级别划分避免数据泄漏

---

## 🙋 常见问题

**Q: 训练时CUDA错误?**
A: 确保PyTorch版本支持sm_120,使用Nightly CUDA 12.8版本

**Q: 如何只检索特定患者?**
A: 修改 `--dicom-root` 指向特定患者文件夹

**Q: 3D Slicer无法打开?**
A: 检查Slicer路径,或手动导入输出文件夹

---

## 👥 作者

LIDC-IDRI肺结节检索系统 - 2025

## 📄 许可

本项目仅供学术研究使用



>>>>>>> 50105a4 (881 project)
