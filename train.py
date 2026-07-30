from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data = "data.yaml",
        epochs = 200,
        imgsz = 960,
        batch = 16,
        device = 0,
        workers = 4,
        patience = 50,
        name = "resistor_v1"
    )

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
