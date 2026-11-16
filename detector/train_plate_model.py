"""Train the plate detector.

   python train_plate_model.py --data data/plates/data.yaml

Needs a GPU for sensible speed. On an RTX 4050 (6 GB) a few hundred to a few thousand images take
roughly half an hour to two hours. The best model is copied to models/plate.pt and its measured
accuracy on the validation images is printed: report those numbers in Chapter 4.
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml from prepare_dataset.py")
    ap.add_argument("--model", default="yolo11s.pt", help="starting weights (yolo11n.pt is smaller and faster)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 picks the largest that fits in GPU memory")
    ap.add_argument("--device", default="0", help='"0" for the first GPU, "cpu" for no GPU')
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="models/plate.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, patience=20, project="runs", name="plate",
                exist_ok=True, plots=True)
    best = str(model.trainer.best)   # Ultralytics says where it saved the best model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    shutil.copy2(best, args.out)

    metrics = YOLO(args.out).val(data=args.data, imgsz=args.imgsz, device=args.device, workers=args.workers)
    box = metrics.box
    print("\nPlate detector results on the validation images")
    print(f"  precision  {box.mp:.3f}")
    print(f"  recall     {box.mr:.3f}")
    print(f"  mAP@0.5    {box.map50:.3f}")
    print(f"  mAP@0.5:95 {box.map:.3f}")
    print(f"\nSaved the model to {args.out}. Set \"plate_model\": \"{args.out}\" in detector_config.json.")


if __name__ == "__main__":
    main()
