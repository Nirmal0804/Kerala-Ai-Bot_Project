"""Quick end-to-end smoke test for the Kerala crop disease pipeline.

This script creates a tiny synthetic dataset with the required 8 class folders,
runs one training epoch, and prints validation accuracy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.training.train_model import train_and_evaluate


CLASS_NAMES = [
    "banana_sigatoka",
    "banana_leaf_spot",
    "banana_healthy",
    "okra_yellow_vein_mosaic",
    "okra_powdery_mildew",
    "okra_healthy",
    "chilli_leaf_curl",
    "chilli_healthy",
]


def _build_tiny_dataset(data_root: Path) -> None:
    """Create a tiny synthetic dataset matching the required class structure."""
    rng = np.random.default_rng(seed=42)
    for class_name in CLASS_NAMES:
        class_dir = data_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            image = rng.integers(low=0, high=256, size=(256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"sample_{i}.jpg"), image)


def main() -> None:
    """Run one-epoch training and print validation accuracy."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir) / "data"
        output_dir = Path(tmp_dir) / "models" / "crop_disease_model"
        _build_tiny_dataset(data_dir)

        result = train_and_evaluate(
            data_dir=str(data_dir),
            save_path=str(output_dir),
            batch_size=4,
            epochs=1,
            val_split=0.25,
            seed=42,
            use_pretrained=False,
        )

        print(f"Validation accuracy: {result['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()