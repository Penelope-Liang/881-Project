import argparse
# from pathlib import Path
from ultralytics import YOLO


def train_yolo_detector(data_yaml, model_name="yolov8s.pt", epochs=50, batch_size=16, 
                       device="cuda", project="plan_b/runs/detect", name="nodule_yolov8"):
    """
    train yolov8 nodule detector
    
    args:
        data_yaml: path to yolo dataset config file
        model_name: pretrained yolov8 model name
        epochs: number of training epochs
        batch_size: batch size
        device: device to use ("cuda" or "cpu")
        project: project directory for saving results
        name: experiment name
    """
    print("="*80)
    print("Training YOLOv8 Nodule Detector")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Data:       {data_yaml}")
    print(f"  Model:      {model_name}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Device:     {device}")
    print(f"  Project:    {project}")
    print(f"  Name:       {name}")
    
    # load pretrained model
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # train
    print("\nStarting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        imgsz=512,
        patience=10,
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    print(f"Results saved to: {project}/{name}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 nodule detector")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO dataset config YAML file")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Pretrained YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--project", type=str, default="plan_b/runs/detect", help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="nodule_yolov8", help="Experiment name")
    
    args = parser.parse_args()
    
    train_yolo_detector(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )

