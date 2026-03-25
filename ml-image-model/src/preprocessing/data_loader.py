"""Load and preprocess plant leaf image datasets for disease detection."""

import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_images(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """Load images and labels from a folder-per-class dataset.

    Args:
        data_dir: Root dataset folder where each subfolder is a class name.

    Returns:
        images: List of raw images loaded with OpenCV.
        labels: Numeric labels aligned with images.
        class_names: Sorted class names where index == label value.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_names = sorted(
        [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    )
    if not class_names:
        raise ValueError(f"No class folders found in: {data_dir}")

    images: List[np.ndarray] = []
    labels: List[int] = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if not os.path.isfile(file_path):
                continue

            image = cv2.imread(file_path)
            if image is None:
                continue

            images.append(image)
            labels.append(label_idx)

    if not images:
        raise ValueError(f"No readable image files found in: {data_dir}")

    return images, np.array(labels, dtype=np.int64), class_names


def preprocess_images(
    images: Sequence[np.ndarray],
    image_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize to 224x224, normalize pixels, and return a NumPy array.

    Args:
        images: Raw images loaded with OpenCV.
        image_size: Desired (width, height) output size.

    Returns:
        Preprocessed image array with shape (N, H, W, C) and float32 dtype.
    """
    processed: List[np.ndarray] = []

    for image in images:
        resized = cv2.resize(image, image_size)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        processed.append(normalized)

    if not processed:
        raise ValueError("No images were provided for preprocessing.")

    return np.array(processed, dtype=np.float32)


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train and validation sets.

    Args:
        images: Preprocessed image array.
        labels: Numeric label array.
        test_size: Validation split fraction.
        random_state: Seed used by train_test_split.

    Returns:
        X_train, X_val, y_train, y_val
    """
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same number of samples.")
    if len(labels) < 2:
        raise ValueError("At least 2 samples are required to split the dataset.")

    unique_labels, counts = np.unique(labels, return_counts=True)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in the range (0.0, 1.0).")

    val_count = int(round(len(labels) * test_size))
    # Stratified splitting requires enough samples in each class and in validation.
    can_stratify = (
        len(unique_labels) > 1
        and np.min(counts) >= 2
        and val_count >= len(unique_labels)
    )
    stratify_labels = labels if can_stratify else None

    return train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )


def load_and_prepare_dataset(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    val_split: float = 0.2,
    random_state: int = 42,
) -> Dict[str, np.ndarray | List[str]]:
    """Convenience wrapper around load, preprocess, and split steps."""
    raw_images, labels, class_names = load_images(data_dir)
    images = preprocess_images(raw_images, image_size=image_size)
    x_train, x_val, y_train, y_val = split_dataset(
        images,
        labels,
        test_size=val_split,
        random_state=random_state,
    )

    return {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
        "class_names": class_names,
    }
