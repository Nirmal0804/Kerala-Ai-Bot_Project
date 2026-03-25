"""Basic tests to validate project structure and imports."""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models import SimpleCNN
from src.preprocessing import preprocess_images


def test_imports_and_construction() -> None:
    model = SimpleCNN(num_classes=2)
    assert model is not None

    sample = np.zeros((300, 300, 3), dtype=np.uint8)
    processed = preprocess_images([sample])
    assert processed.shape == (1, 224, 224, 3)
