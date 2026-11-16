"""Train the vehicle detector.

   python train_vehicle_model.py --data data/vehicles/data.yaml

Needs a GPU for sensible speed. On an RTX 4050 (6 GB) a few hundred to a few thousand images take
roughly half an hour to two hours. The best model is copied to models/vehicle.pt and its measured
accuracy on the validation images is printed: report those numbers in Chapter 4.
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml from prepare_vehicles.py")
    ap.add_argument("--model", default="yolo11s.pt", help="starting weights (yolo11n.pt is smaller and faster)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=-1, help="-1 picks the largest that fits in GPU memory")
    ap.add_argument("--device", default="0", help='"0" for the first GPU, "cpu" for no GPU')
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="models/vehicle.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, patience=20, project="runs", name="vehicle",
                exist_ok=True, plots=True)
    best = str(model.trainer.best)   # Ultralytics says where it saved the best model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    shutil.copy2(best, args.out)

    metrics = YOLO(args.out).val(data=args.data, imgsz=args.imgsz, device=args.device, workers=args.workers)
    box = metrics.box
    print("\nVehicle detector results on the validation images")
    print(f"  precision  {box.mp:.3f}")
    print(f"  recall     {box.mr:.3f}")
    print(f"  mAP@0.5    {box.map50:.3f}")
    print(f"  mAP@0.5:95 {box.map:.3f}")
    names = metrics.names
    print("  per class mAP@0.5:", ", ".join(f"{names[int(c)]} {v:.2f}" for c, v in zip(metrics.ap_class_index, box.ap50)))
    print(f"\nSaved the model to {args.out}. In detector_config.json set \"model\": \"{args.out}\" and \"vehicle_classes\": [].")


if __name__ == "__main__":
    main()
