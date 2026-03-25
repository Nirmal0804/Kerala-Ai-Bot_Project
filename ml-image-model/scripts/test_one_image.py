"""Run single-image disease prediction using the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_disease, load_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a single image with trained crop disease model")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--model",
        default="models/crop_disease_model",
        help="Path to trained model directory or .keras file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load class names saved with the model
    try:
        class_names = load_class_names(args.model)
    except Exception as e:
        print(f"Warning: Could not load class names: {e}")
        class_names = None
    
    result = predict_disease(
        image_path=str(image_path),
        model_path=args.model,
        class_names=class_names,
    )

    print(f"Image: {image_path}")
    if class_names:
        print(f"Class order: {class_names}")
    print(f"Prediction: {result['crop']}_{result['disease']}")
    print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    main()
