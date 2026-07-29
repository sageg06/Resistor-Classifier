"""
Real time resistor classifier

Streams webcam frames to a Roboflow-hosted object detection model,
annotates the highest confidence detection on screen, then reports the detected
resistor value to an Arduino over serial for display.
"""
# --- Imports ---
import os
import sys
import time

import cv2
import serial
from inference_sdk import InferenceHTTPClient

# --- Configuration ---
ARDUINO_PORT= os.environ.get("ARDUINO_PORT", "COM3")
BAUD_RATE = 9600

ROBOFLOW_API_URL= "https://serverless.roboflow.com"
MODEL_ID= "resistor-detector-rtlor/7"

CAMERA_INDEX = 0
INFERENCE_INTERVAL = 10 # Infers every Nth frame, increase this if webcam feed is slow/choppy
CONFIDENCE_THRESHOLD = 0.5

BOX_COLOR = (0, 255, 0) #BGR not RGB
WARNING_COLOR = (0, 0, 255)
WINDOW_NAME = "Resistor Detector"

# --- Setup Helpers ---
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
def best_guess(prediction, threshold):
    #Returns the highest confidence prediction above 'threshold'
    confident = [p for p in prediction if p.get("confidence", 0) > threshold]
    if not confident:
        return None
    return max(confident, key=lambda p: p["confidence"])

def draw_box(frame, prediction, label):
    # Draws bounding box and label for one prediction
    # Roboflow reports box center and size, OpenCV wants opposite corners
    x, y = prediction["x"], prediction["y"]
    width, height = prediction["width"], prediction["height"]
    top_left = (int(x - width / 2), int(y - height / 2))
    bottom_right = (int(x + width / 2), int(y + height / 2))

    cv2.rectangle(frame, top_left, bottom_right, BOX_COLOR, 2)
    cv2.putText(
        frame,
        f"{label} ({prediction['confidence']:.2f})",
        (top_left[0], max(top_left[1] - 10, 20)),  # Keep text on screen.
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        BOX_COLOR,
        2,
    )

def classify_frame(client, frame):
    # Runs inference on one frame
    # returns (annotated_frame, label) and label is unknown when nothing clears confidence threshold
    try:
        result = client.infer(frame, model_id=MODEL_ID)
        predictions = result.get("predictions", [])
    except Exception as exc:
        print(f"Inference error: {exc}")
        predictions = []

    annotated = frame.copy()
    best = best_guess(predictions, CONFIDENCE_THRESHOLD)

    if best is None:
        cv2.putText(
            annotated, "No resistor detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, WARNING_COLOR, 2,
        )
        return annotated, "Unknown"

    label = best.get("class", "Unknown").replace("_", " ")
    draw_box(annotated, best, label)
    return annotated, label

def send_label(arduino, label):
    # Sends a newline terminated label, the sketch will read until '\\n'
    message = f"{label}\n" if label != "Unknown" else "NONE\n"
    arduino.write(message.encode("utf-8"))
    print(f"Sent to Arduino: {label}")

# --- Main Loop ---
def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY not set! See README for setup!")
        exit()

    client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=api_key)

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
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame :(")
                break

            frame_count += 1
            if frame_count % INFERENCE_INTERVAL == 0:
                display_frame, label = classify_frame(client, frame)
                if label != last_sent_label:
                    send_label(arduino, label)
                    last_sent_label = label
            else:
                display_frame = frame

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
