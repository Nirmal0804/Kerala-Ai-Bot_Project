# pyright: reportMissingImports=false, reportMissingModuleSource=false

"""Train a MobileNetV2 model for crop disease classification."""

import argparse
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical

from src.preprocessing import load_and_prepare_dataset


IMAGE_SIZE: tuple[int, int] = (224, 224)
NUM_CLASSES: int = 8
BATCH_SIZE: int = 32
EPOCHS: int = 25


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Train crop disease detection model")
	parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root folder")
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
	parser.add_argument("--epochs", type=int, default=EPOCHS)
	parser.add_argument("--val-split", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--save-path",
		type=str,
		default="models/crop_disease_model",
		help="Output path for saved model",
	)
	parser.add_argument(
		"--no-pretrained",
		action="store_true",
		help="Disable ImageNet pretrained weights (useful for fast local tests).",
	)
	return parser.parse_args()


def build_model(num_classes: int = NUM_CLASSES, use_pretrained: bool = True) -> Model:
	"""Create MobileNetV2-based classifier."""
	if num_classes < 2:
		raise ValueError("num_classes must be >= 2")

	base_model = MobileNetV2(
		input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
		include_top=False,
		weights="imagenet" if use_pretrained else None,
	)
	base_model.trainable = False

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(256, activation="relu")(x)
	x = Dropout(0.3)(x)
	outputs = Dense(num_classes, activation="softmax")(x)

	model = Model(inputs=base_model.input, outputs=outputs)
	model.compile(
		optimizer=Adam(),
		loss="categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def build_augmented_dataset(
	x: np.ndarray | tf.Tensor,
	y: np.ndarray | tf.Tensor,
	batch_size: int,
	training: bool,
) -> tf.data.Dataset:
	"""Build tf.data pipeline and apply augmentation to training data."""
	if batch_size < 1:
		raise ValueError("batch_size must be >= 1")

	dataset = tf.data.Dataset.from_tensor_slices((x, y))

	if training:
		augmentation = keras.Sequential(
			[
				tf.keras.layers.RandomFlip("horizontal"),
				tf.keras.layers.RandomRotation(0.1),
				tf.keras.layers.RandomZoom(0.1),
			]
		)

		dataset = dataset.shuffle(buffer_size=max(int(len(x)), 1), seed=42)
		dataset = dataset.map(
			lambda image, label: (augmentation(image, training=True), label),
			num_parallel_calls=tf.data.AUTOTUNE,
		)

	dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	return dataset


def train_and_evaluate(
	*,
	data_dir: str,
	save_path: str = "models/crop_disease_model",
	batch_size: int = BATCH_SIZE,
	epochs: int = EPOCHS,
	val_split: float = 0.2,
	seed: int = 42,
	use_pretrained: bool = True,
) -> dict[str, Any]:
	"""Run full training pipeline and return model/metrics metadata."""
	tf.random.set_seed(seed)
	np.random.seed(seed)

	data = load_and_prepare_dataset(
		data_dir=data_dir,
		image_size=IMAGE_SIZE,
		val_split=val_split,
		random_state=seed,
	)

	class_names = data["class_names"]
	if len(class_names) != NUM_CLASSES:
		raise ValueError(
			f"Expected exactly {NUM_CLASSES} classes, found {len(class_names)}: {class_names}"
		)

	x_train = np.asarray(data["x_train"], dtype=np.float32)
	x_val = np.asarray(data["x_val"], dtype=np.float32)
	y_train = to_categorical(np.asarray(data["y_train"]), num_classes=NUM_CLASSES)
	y_val = to_categorical(np.asarray(data["y_val"]), num_classes=NUM_CLASSES)

	train_dataset = build_augmented_dataset(x_train, y_train, batch_size=batch_size, training=True)
	val_dataset = build_augmented_dataset(x_val, y_val, batch_size=batch_size, training=False)

	model = build_model(num_classes=NUM_CLASSES, use_pretrained=use_pretrained)

	print("Starting training...")
	history = model.fit(
		train_dataset,
		validation_data=val_dataset,
		epochs=epochs,
		verbose=1,
	)

	print("Evaluating model...")
	loss, accuracy = model.evaluate(val_dataset, verbose=1)
	print(f"Validation loss: {loss:.4f}")
	print(f"Validation accuracy: {accuracy:.4f}")

	output_path = Path(save_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Keras 3 requires an explicit extension for model.save(...).
	if output_path.suffix in {".keras", ".h5"}:
		final_model_path = output_path
	else:
		output_path.mkdir(parents=True, exist_ok=True)
		final_model_path = output_path / "model.keras"

	model.save(final_model_path)
	print(f"Model saved to: {final_model_path}")

	return {
		"model": model,
		"history": history.history,
		"val_loss": float(loss),
		"val_accuracy": float(accuracy),
		"class_names": class_names,
		"save_path": str(final_model_path),
	}


def main() -> None:
	"""CLI entrypoint for training."""
	args = parse_args()
	train_and_evaluate(
		data_dir=args.data_dir,
		save_path=args.save_path,
		batch_size=args.batch_size,
		epochs=args.epochs,
		val_split=args.val_split,
		seed=args.seed,
		use_pretrained=not args.no_pretrained,
	)


if __name__ == "__main__":
	main()
