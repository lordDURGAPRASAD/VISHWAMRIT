import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# Configuration Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "../dataset"  # Folder containing subfolders for each leaf class

def train():
    print("Starting Model Training Pipeline...")

    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"[Warning] Dataset directory '{DATASET_DIR}' not found.")
        print("Creating placeholder class_names.json for fallback demonstration...")
        default_classes = ["Neem", "Tulsi", "Aloe_Vera", "Brahmi", "Mint"]
        with open("model/class_names.json", "w") as f:
            json.dump(default_classes, f)
        print("Saved placeholder classes to model/class_names.json")
        return

    # 1. Load Training Dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # 2. Load Validation Dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Extract and save class names
    class_names = train_ds.class_names
    os.makedirs("model", exist_ok=True)
    with open("model/class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"Extracted {len(class_names)} classes: {class_names}")

    # 3. Data Augmentation Pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    # 4. Base Transfer Learning Model (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base layers

    # 5. Assemble Classifier Model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # 6. Compile Model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 7. Train Model
    print("Fitting model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 8. Save Trained Model Weights
    model.save("model/vishwamrit_model.h5")
    print("Model training complete! Saved to model/vishwamrit_model.h5")

if __name__ == "__main__":
    train()