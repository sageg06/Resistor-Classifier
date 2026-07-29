# Resistor-Classifier
Real-time resistor identification from a webcam feed with results pushed to an Arduino over serial for on-device display.

A webcam frame is sent to a custom trained object detection model hosted on Roboflow. The highest confidence detection is drawn on a live preview window and its class label (ex. 220 Ohm) is written to an Arduino, which displays the value on a connected LCD screen. Serial writes are debounced so the board is only updated when the detected value actually changes.

<TODO: add picture of preview window>

## Hardware
