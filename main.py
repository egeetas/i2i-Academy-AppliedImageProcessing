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


if __name__ == "__main__":
    main()
