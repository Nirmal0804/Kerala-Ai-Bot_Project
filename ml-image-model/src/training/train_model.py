import os
from pathlib import Path

import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam


def build_model(num_classes: int) -> Model:
    """Build transfer learning model using MobileNetV2."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom head
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model




def train_and_evaluate(
    data_dir: str = "Datasets/train",
    val_data_dir: str = "Datasets/valid",
    save_path: str = "models/crop_disease_model",
    batch_size: int = 32,
    epochs: int = 5,
    val_split: float = 0.2,
    seed: int = 42,
    use_pretrained: bool = True,
):
    """Train model on dataset."""
    # Set seed for reproducibility
    tf.random.set_seed(seed)
    
    # Load training data (before any transformations that lose class_names)
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    raw_train_data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        seed=seed,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="categorical"
    )
    
    # Capture class names before applying transformations
    class_names = raw_train_data.class_names
    num_classes = len(class_names)
    
    # Apply normalization
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_data = raw_train_data.map(lambda x, y: (normalization_layer(x), y))
    
    # Load validation data if available, otherwise split from training
    val_data = None
    if Path(val_data_dir).exists():
        raw_val_data = tf.keras.utils.image_dataset_from_directory(
            Path(val_data_dir),
            seed=seed,
            image_size=(224, 224),
            batch_size=batch_size,
            label_mode="categorical"
        )
        val_data = raw_val_data.map(lambda x, y: (normalization_layer(x), y))
    else:
        # Split training data
        train_size = int(len(train_data) * (1 - val_split))
        train_data_split = train_data.take(train_size)
        val_data_split = train_data.skip(train_size)
        train_data = train_data_split
        val_data = val_data_split
    
    # Build model
    model = build_model(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # Train
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=1
    )
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/model.keras")
    
    # Save class names
    import json
    class_names_path = f"{save_path}/class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)
    
    print(f"Model saved to {save_path}/model.keras")
    print(f"Class names saved to {class_names_path}")
    
    val_accuracy = history.history["val_accuracy"][-1] if "val_accuracy" in history.history else 0.0
    return {"val_accuracy": val_accuracy, "class_names": class_names}


def main() -> None:
    """Main entry point for training."""
    result = train_and_evaluate()
    print(f"Training complete. Validation accuracy: {result['val_accuracy']:.4f}")


if __name__ == "__main__":
    main() 