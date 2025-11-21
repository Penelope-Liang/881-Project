# Step 6 - YOLOv8 Nodule Detector Training

- Scripts: train_detector.py
- What it does:
  - Train YOLOv8 model to detect lung nodules on full CT slices
  - Generate trained model checkpoints and training metrics
- How to run:

```bash
# quick test for 1 epoch, check if it works or not
python step6/train_detector.py \
    --data plan_b/detection_data/nodule_detection.yaml \
    --model yolov8s.pt \
    --epochs 1 \
    --batch 2 \
    --device cpu

# full training for 50 epochs
python step6/train_detector.py \
    --data plan_b/detection_data/nodule_detection.yaml \
    --model yolov8s.pt \
    --epochs 50 \
    --batch 16 \
    --device cpu
```

See main README.md for detailed instructions and examples.
