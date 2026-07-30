"""
Real time resistor classifier

Runs a locally trained YOLOv8 model on webcam franes, annotates highest confidence
detection on screen, then reports the detected resistor value to an Arduino over serial for display
"""
# --- Imports ---
import os
import sys
import time

import cv2
import serial
from ultralytics import YOLO
# --- Configuration ---
ARDUINO_PORT= os.environ.get("ARDUINO_PORT", "COM3")
BAUD_RATE = 9600

MODEL_PATH = "weights/best.pt"
INFERENCE_SIZE = 960

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5

BOX_COLOR = (0, 255, 0) #BGR not RGB
WARNING_COLOR = (0, 0, 255)
WINDOW_NAME = "Resistor Detector"

# --- Setup Helpers ---
def load_model(path):
    #Loads trained weights
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}, try copying best.pt from runs/detect/")
    return YOLO(path)

def connect_to_arduino(port, baud):
    # Open the serial link and wait on board to reset
    arduino = serial.Serial(port, baud, timeout=1)
    time.sleep(2)
    return arduino

def open_camera(index):
    # Opens webcam
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not(cap.isOpened()):
        print("Error! Could not open camera")
    return cap

# --- Inference and Drawing ---
def best_guess(results):
    #Returns the highest confidence prediction above 'threshold'
    boxes = results.boxes
    if len(boxes) == 0:
        return None
    i = boxes.conf.argmax().item()
    corners = boxes.xyxy[i].int().tolist()
    return corners, boxes.conf[i].item(), int(boxes.cls[i].item())

def draw_box(frame, corners, confidence, label):
    # Draws bounding box and label for one prediction
    x1, y1, x2, y2 = corners

    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (x1, max(y1 - 10, 20)),  # Keep text on screen.
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        BOX_COLOR,
        2,
    )

def classify_frame(model, frame):
    # Runs inference on one frame
    # returns (annotated_frame, label) and label is unknown when nothing clears confidence threshold
    try:
        results = model(
            frame,
            imgsz=INFERENCE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]
    except Exception as exc:
        print(f"Inference error: {exc}")
        return frame.copy(), "Unknown"

    annotated = frame.copy()
    best = best_guess(results)

    if best is None:
        cv2.putText(
            annotated, "No resistor detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, WARNING_COLOR, 2,
        )
        return annotated, "Unknown"

    corners, confidence, class_index = best
    label = model.names[class_index].replace("_", " ")
    draw_box(annotated, corners, confidence, label)
    return annotated, label

def send_label(arduino, label):
    # Sends a newline terminated label, the sketch will read until '\\n'
    message = f"{label}\n" if label != "Unknown" else "NONE\n"
    arduino.write(message.encode("utf-8"))
    print(f"Sent to Arduino: {label}")

# --- Main Loop ---
def main():
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Classes: {', '.join(model.names.values())}")

    try:
        arduino = connect_to_arduino(ARDUINO_PORT, BAUD_RATE)
    except serial.SerialException as exc:
        print(f"Could not connect to Arduino on {ARDUINO_PORT}: {exc}")
        exit()
    print(f"Arduino connected on {ARDUINO_PORT}!!")

    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as exc:
        arduino.close()
        print(f"Failed to open camera: {exc}, try a different camera index :(")
        exit()
    print("Camera successfully opened! Press Q to quit.")

    last_sent_label = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame :(")
                break

            display_frame, label = classify_frame(model, frame)
            if label != last_sent_label:
                send_label(arduino, label)
                last_sent_label = label

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        arduino.write(b"NONE\n") # Clears display while exiting
        arduino.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Goodbye! :)")

if __name__ == "__main__":
    main()
