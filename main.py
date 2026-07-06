"""Automated License Plate Recognition application entry point."""

import argparse
from pathlib import Path

import cv2


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


if __name__ == "__main__":
    main()
