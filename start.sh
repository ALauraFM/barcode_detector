#!/bin/bash


# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

# Install required Python packages
pip install opencv-python pyzbar

#Install pyrealsense2
pip install pyrealsense2

# Run the Python script
python3 /src/barcode_reader.py  


# Exit the script
exit 0