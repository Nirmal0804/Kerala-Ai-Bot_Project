"""Run prediction for every image in the uploads folder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_disease, load_class_names

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_upload_images(upload_dir: Path) -> list[Path]:
    if not upload_dir.is_dir():
        raise FileNotFoundError(f"Upload directory not found: {upload_dir}")

    images = [p for p in sorted(upload_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise ValueError(
            f"No supported images found in {upload_dir}. "
            "Upload images with .jpg/.jpeg/.png/.bmp/.webp extensions."
        )

    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions for all uploaded images")
    parser.add_argument(
        "--upload-dir",
        default="uploads",
        help="Folder containing uploaded test images",
    )
    parser.add_argument(
        "--model",
        default="models/crop_disease_model",
        help="Path to trained model directory or .keras file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    upload_dir = Path(args.upload_dir)
    image_paths = _list_upload_images(upload_dir)
    
    # Load class names saved with the model
    try:
        class_names = load_class_names(args.model)
    except Exception as e:
        print(f"Warning: Could not load class names: {e}")
        class_names = None

    print(f"Found {len(image_paths)} image(s) in: {upload_dir}")
    if class_names:
        print(f"Class order: {class_names}")

    for image_path in image_paths:
        result = predict_disease(
            image_path=str(image_path),
            model_path=args.model,
            class_names=class_names,
        )
        print("-" * 50)
        print(f"Image: {image_path}")
        print(f"Prediction: {result['crop']}_{result['disease']}")
        print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    main()
