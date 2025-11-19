# Step 5 - Contrastive Retrieval (Image/Text)

This step trains CLIP-style encoders to align ROI crops with size-related text prompts. It supports Top‑K retrieval and can be fused with the rule-based diameter filter.

## Folder
- Code: `step5/`
- Outputs (suggested): `outputs/step5/`

## Text prompts (curated)
- Size templates:
  - "a lung nodule with diameter <= 6 mm"
  - "a lung nodule with diameter 6-10 mm"
  - "a lung nodule with diameter > 10 mm"
  - "a lung nodule with diameter > 3 mm"
- Short synonyms:
  - "a small lung nodule"
  - "a medium-size lung nodule"
  - "a large lung nodule"
- Query examples for retrieval:
  - ">=3 mm", "<=6 mm", "6-10 mm", ">10 mm"
  - "a lung nodule with diameter > 6 mm"
  - "a large lung nodule"

## 1) Train (patient-level train/val)
```powershell
# Use the same splits produced in Step 4
python -m step5.train_clip ^
  --roi-csv outputs/step2/roi_with_diam.csv ^
  --train-patients step4/splits/train_patients.txt ^
  --val-patients step4/splits/val_patients.txt ^
  --epochs 5 --bs 256 --img-size 256 --emb 128 ^
  --out outputs/step5
```
- Output: `outputs/step5/clip_best.pth` (contains image encoder, text encoder, vocab, cfg)

## 2) Build image embeddings (test split)
```powershell
python -m step5.build_embeddings ^
  --roi-csv outputs/step2/roi_with_diam.csv ^
  --patients-file step4/splits/test_patients.txt ^
  --model outputs/step5/clip_best.pth ^
  --out-dir outputs/step5/emb_test ^
  --img-size 256 --emb 128
```
- Output: `emb_test/img_embeds.npy`, `emb_test/meta.parquet`

## 3) Text → Top‑K retrieval
```powershell
python -m step5.semantic_query ^
  --emb-dir outputs/step5/emb_test ^
  --model outputs/step5/clip_best.pth ^
  --query "a lung nodule with diameter > 6 mm" ^
  --topk 10 ^
  --out outputs/step5/topk_demo
```
- Output: `outputs/step5/topk_demo/topk.csv` (dicom_path, diameter_mm, similarity)

## Notes
- If CUDA fails on your Windows setup, training or embedding can run on CPU (slower but fine for demo).
- You can fuse retrieval score with rule-based diameter score at application level: `alpha*cosine + (1-alpha)*rule`.