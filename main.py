"""Automated License Plate Recognition application entry point."""

import argparse
import re
import warnings
from pathlib import Path

import cv2
import easyocr


def load_image(image_path: Path):
    """Load an image from disk or raise a clear error."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image could not be loaded: {image_path}")
    return image


def save_image(image, output_path: Path) -> None:
    """Save a processing result or raise a clear error."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Image could not be saved: {output_path}")


def convert_to_grayscale(image):
    """Convert a BGR image to a single-channel grayscale image."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(grayscale):
    """Reduce small image details with a 5x5 Gaussian filter."""
    return cv2.GaussianBlur(grayscale, (5, 5), 0)


def detect_edges(blurred):
    """Detect strong intensity transitions with the Canny algorithm."""
    return cv2.Canny(blurred, 50, 150)


def find_plate_bounds(edges):
    """Select the most plate-like contour using geometry and image position."""
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    connected_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closing_kernel)
    contours, _ = cv2.findContours(
        connected_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    image_height, image_width = edges.shape
    image_center_x = image_width / 2
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if height == 0:
            continue

        aspect_ratio = width / height
        center_distance = abs((x + width / 2) - image_center_x) / image_center_x
        is_plate_sized = (
            area >= 500
            and 0.08 * image_width <= width <= 0.35 * image_width
            and 0.04 * image_height <= height <= 0.20 * image_height
        )
        is_plate_shaped = 1.5 <= aspect_ratio <= 3.0
        is_in_lower_half = y >= 0.50 * image_height

        if is_plate_sized and is_plate_shaped and is_in_lower_half:
            position_weight = max(0.1, 1 - center_distance) ** 4
            candidates.append((area * position_weight, (x, y, width, height)))

    if not candidates:
        raise RuntimeError("No license plate contour was found.")

    return max(candidates, key=lambda candidate: candidate[0])[1], len(contours)


def crop_with_padding(image, bounds):
    """Crop a detected plate while preserving a small border around it."""
    x, y, width, height = bounds
    image_height, image_width = image.shape[:2]
    padding_x = round(width * 0.03)
    padding_y = round(height * 0.10)
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(image_width, x + width + padding_x)
    y2 = min(image_height, y + height + padding_y)
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def recognize_plate(plate):
    """Read uppercase Latin letters and digits from a cropped plate."""
    warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
    reader = easyocr.Reader(
        ["en"], gpu=False, model_storage_directory=".easyocr", verbose=False
    )
    results = reader.readtext(
        plate,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    if not results:
        raise RuntimeError("OCR could not recognize text on the license plate.")

    _, text, confidence = max(results, key=lambda result: result[2])
    cleaned_text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not cleaned_text:
        raise RuntimeError("OCR returned no valid license plate characters.")
    return cleaned_text, float(confidence)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect and read a vehicle plate.")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path("images/car.jpg"),
        help="Path to the vehicle image (default: images/car.jpg)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the ALPR processing pipeline."""
    args = parse_args()
    image = load_image(args.image)
    height, width = image.shape[:2]

    print(f"Image loaded: {args.image}")
    print(f"Image size: {width}x{height}")

    grayscale = convert_to_grayscale(image)
    grayscale_path = Path("output/01_grayscale.jpg")
    save_image(grayscale, grayscale_path)
    print(f"Grayscale image saved: {grayscale_path}")

    blurred = reduce_noise(grayscale)
    blurred_path = Path("output/02_blurred.jpg")
    save_image(blurred, blurred_path)
    print(f"Blurred image saved: {blurred_path}")

    edges = detect_edges(blurred)
    edges_path = Path("output/03_edges.jpg")
    save_image(edges, edges_path)
    print(f"Edge image saved: {edges_path}")

    plate_bounds, contour_count = find_plate_bounds(edges)
    plate, padded_bounds = crop_with_padding(image, plate_bounds)
    x, y, width, height = padded_bounds

    detected = image.copy()
    cv2.rectangle(detected, (x, y), (x + width, y + height), (0, 255, 0), 2)
    detected_path = Path("output/04_detected_plate.jpg")
    plate_path = Path("output/05_cropped_plate.jpg")
    save_image(detected, detected_path)
    save_image(plate, plate_path)

    print(f"Contours found: {contour_count}")
    print(f"Plate bounds: x={x}, y={y}, width={width}, height={height}")
    print(f"Detected plate image saved: {detected_path}")
    print(f"Cropped plate image saved: {plate_path}")

    plate_text, confidence = recognize_plate(plate)
    print(f"Recognized license plate: {plate_text}")
    print(f"OCR confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
