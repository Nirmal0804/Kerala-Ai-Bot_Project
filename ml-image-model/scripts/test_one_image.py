"""Run single-image disease prediction using the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_disease


def _discover_class_names(dataset_train_dir: Path) -> list[str]:
    """Discover class names from training dataset folder names in sorted order."""
    if not dataset_train_dir.is_dir():
        raise FileNotFoundError(f"Training dataset directory not found: {dataset_train_dir}")

    class_names = sorted([p.name for p in dataset_train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class directories found in: {dataset_train_dir}")

    return class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a single image with trained crop disease model")
    parser.add_argument("image_path", help="Path to input image")
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

    image_path = Path(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    class_names = _discover_class_names(Path(args.dataset_train_dir))
    result = predict_disease(
        image_path=str(image_path),
        model_path=args.model,
        class_names=class_names,
    )

    print(f"Image: {image_path}")
    print(f"Class order: {class_names}")
    print(f"Prediction: {result['crop']}_{result['disease']}")
    print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    main()
