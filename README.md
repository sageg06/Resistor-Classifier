# Resistor-Classifier
Real-time resistor identification from a webcam feed with results pushed to an Arduino over serial for on-device display.

A webcam frame is sent to a custom trained object detection model hosted on Roboflow. The highest confidence detection is drawn on a live preview window and its class label (ex. 220 Ohm) is written to an Arduino, which displays the value on a connected LCD screen. Serial writes are debounced so the board is only updated when the detected value actually changes.

<TODO: add picture of preview window>

## Hardware
* Webcam (built-in or USB)
* Arduino UNO connected over USB
* LCD Display attached to the Arduino

## Software Requirements
* Python 3.9+
* A Roboflow account and API key
* Dependencies in requirements.txt

## Setup
```
git clone https://github.com/sageg06/resistor-classifier.git
cd resistor-classifier
python -m venv .venv
source .venv/bin/activate        Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Flash arduino/resistor_display.ino to the board before running the Python side.

## Configuration
Both settings are read from environment variables, no keys are stored in the source
```
# macOS / Linux:
export ROBOFLOW_API_KEY="your_key_here"
export ARDUINO_PORT="/dev/ttyACMO"
```
```
# Windows (PowerShell):
$env:ROBOFLOW_API_KEY="your_key_here"
$env:ARDUINO_PORT="COM#"
```
replace the "#" with your port number. You can find your port number in the Arduino IDE under Tools -> Port.

## Usage
```
python resistor_classifier.py
```
A preview window opens showing the camera feed with detections overlaid. Press **Q** to quit, upon exit the script sends 'NONE' to the Arduino to clear the display and release the camera.

## How It Works
| Stage | What Happens |
| --- | --- |
| Capture | OpenCV reads frames from the webcam in a loop |
| Throttle | Only every 10th frame is sent for inference to stay within API rate limits and keep preview from being too choppy |
| Inference | The frame is sent to Roboflow servers running the model `resistor-detector-rtlor/8` |
| Filter | Predictions below 50% confidence are discarded, highest confidence surviving guess wins |
| Annotate | Roboflow returns box center + dimensions which are converted to corner coordinates for `cv2.rectangle` |
| Report | The class label is written to serial only when it differs from the last value sent |

Tuning knobs are constants at the top of `resistor_classifier.py` for ease of access!

### Serial Protocol
The python side writes ASCII with newline termination 
```
220 Ohm\n   # If detected
NONE\n    # Nothing detected
```
The Arduino sketch reads until `\n` and updates the display with the result.

## Model
Trained on a custom datasheet of through-hole resistors from the Arduino beginner kit captured under varied lighting and orientation.
* Trained on 7 classes: 10k_Ohm, 10M_Ohm, 1k_Ohm, 1M_Ohm, 220_Ohm, 47k_Ohm, 560_Ohm
* 80/10/10 train/test/validate split
* 94.2% mAP@50, 94.1% Precision, 85.8% Recall
* <TODO: add link to Roboflow page?>

## Known Limitations 
- Bounding boxes are only drawn on inference frames, so the overlay flickers at the throttle rate rather than persisting between inferences
- Only the single highest confidence detection is reported so multiple resistors in frame aren't handled
- Detection quality degrades under uneven lighting and odd angles
- Requires network access since inference is hosted rather than local

## Possible Extensions 
- Cache and redraw the last annotation on skipped frames for a stable overlay
- Run model locally via `inference` to remove the network dependency
- Report all detections and their positions rather than just the best one

