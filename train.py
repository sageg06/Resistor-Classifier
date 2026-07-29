from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data = "data.yaml",
        epochs = 3,
        imgsz = 640,
        batch = 32,
        device = 0,
        workers = 8,
        patience = 25,
        project = "runs",
        name = "resistor_v1"
    )

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
