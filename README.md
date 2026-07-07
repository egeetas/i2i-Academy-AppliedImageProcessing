# i2i-Academy-AppliedImageProcessing

An Automated License Plate Recognition (ALPR) exercise implemented with
classical image-processing techniques and OCR.

## Pipeline

1. Load the vehicle image with OpenCV.
2. Convert it to grayscale.
3. Reduce noise with a Gaussian filter.
4. Detect edges with the Canny algorithm.
5. Find and score plate-like contours.
6. Crop the selected plate region.
7. Recognize its text with EasyOCR.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.13 is used for the verified development environment. EasyOCR downloads
its model files into `.easyocr/` on the first run.

## Usage

Place a clear vehicle photograph at `images/car.jpg`, then run:

```bash
python main.py
```

An alternative input and output directory can be supplied explicitly:

```bash
python main.py path/to/vehicle.jpg --output-dir output
```

The recognized plate and OCR confidence are printed to the terminal. Processing
artifacts are written in order:

- `01_grayscale.jpg`
- `02_blurred.jpg`
- `03_edges.jpg`
- `04_detected_plate.jpg`
- `05_cropped_plate.jpg`

Input images, generated output, OCR models, recordings, screenshots, and the
assignment document are intentionally excluded from the repository.
