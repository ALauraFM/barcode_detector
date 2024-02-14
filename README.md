# Barcode Reader using python
This is a Python script to decode barcodes from images captured by a camera. It uses the OpenCV library for image processing and the pyzbar library for barcode decoding.

## Functionality

The script uses a camera connected to the computer to capture real-time images. It detects barcodes in these images and decodes them, displaying the result on the screen.

The barcode decoding process follows these steps:

1. Capture of the image from the camera.
2. Pre-processing of the image for edge detection.
3. Search for rectangles with paralel lines
4. Finds the rectangles that fits in the parameters
5. Sending the rectangle to the pyzbar function.
6. Display of the result on the screen.

## How to Use

To use the barcode decoder code, follow these steps:

1. Clone or download this repository to your computer
2. Ensure you have a realsense camera connected to the computer.
3. Run the `start.sh` script to start the decoder, as it follows:

```
./start.sh
```
The start.sh script contains the installation of all the prerequisites recquired to run the code
