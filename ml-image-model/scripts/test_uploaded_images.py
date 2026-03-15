"""Run prediction for every image in the uploads folder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_disease

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _discover_class_names(dataset_train_dir: Path) -> list[str]:
    if not dataset_train_dir.is_dir():
        raise FileNotFoundError(f"Training dataset directory not found: {dataset_train_dir}")

    class_names = sorted([p.name for p in dataset_train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class directories found in: {dataset_train_dir}")

    return class_names


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
        default="models/crop_disease_model.keras",
        help="Path to trained model file (.keras)",
    )
    parser.add_argument(
        "--dataset-train-dir",
        default="Datasets/train",
        help="Training dataset folder used to infer class order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    upload_dir = Path(args.upload_dir)
    image_paths = _list_upload_images(upload_dir)
    class_names = _discover_class_names(Path(args.dataset_train_dir))

    print(f"Found {len(image_paths)} image(s) in: {upload_dir}")
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
