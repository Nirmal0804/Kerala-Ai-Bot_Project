"""Disease prediction utilities for leaf images using TensorFlow models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)

# Class order must match the order used at training time.
DEFAULT_CLASS_NAMES: List[str] = [
    "banana_sigatoka",
    "banana_leaf_spot",
    "banana_healthy",
    "okra_yellow_vein_mosaic",
    "okra_powdery_mildew",
    "okra_healthy",
    "chilli_leaf_curl",
    "chilli_healthy",
]


def load_trained_model(model_path: str = "models/crop_disease_model") -> tf.keras.Model:
    """Load a trained TensorFlow model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")

    if path.is_dir():
        candidate = path / "model.keras"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Expected model file not found: {candidate}. "
                "Train the model first or provide a direct .keras/.h5 model path."
            )
        return tf.keras.models.load_model(candidate)

    return tf.keras.models.load_model(path)


def preprocess_image(image_path: str, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Read image from path, resize to 224x224, normalize, and add batch dimension."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def _split_class_name(class_name: str) -> tuple[str, str]:
    """Split class label into crop and disease components."""
    if "_" not in class_name:
        return class_name, "unknown"
    crop, disease = class_name.split("_", 1)
    return crop, disease


def predict_disease(
    image_path: str,
    model_path: str = "models/crop_disease_model",
    class_names: List[str] | None = None,
) -> Dict[str, float | str]:
    """Predict crop disease from a leaf image and return result with confidence.

    Returns:
        Example:
            {
                "crop": "banana",
                "disease": "sigatoka",
                "confidence": 0.91,
            }
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    model = load_trained_model(model_path)
    image_batch = preprocess_image(image_path, image_size=IMAGE_SIZE)

    predictions = model.predict(image_batch, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    confidence = float(predictions[pred_idx])

    class_name = class_names[pred_idx]
    crop, disease = _split_class_name(class_name)

    return {
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence, 4),
    }
