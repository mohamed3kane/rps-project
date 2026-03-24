# retrain_and_convert.py
# Trains a compact CNN on the rock-paper-scissors image dataset collected from
# the ESP32-S3 camera, evaluates accuracy, saves the Keras model, and converts
# it to TFLite format for potential deployment.
# Written with assistance from Claude (Anthropic, claude-sonnet-4-6)

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# --- Settings ---
DATASET_DIR = os.path.expanduser("~/Desktop/dataset")  # Root folder containing rock/, paper/, scissors/
IMG_SIZE = 32       # Images are resized to 32x32 to match ESP preprocessing
EPOCHS = 40         # Number of full passes through the training data
BATCH_SIZE = 32     # Number of images processed per gradient update
CLASSES = ["rock", "paper", "scissors"]  # Class order determines label indices (0, 1, 2)

# --- Load Dataset ---
def load_dataset():
    # Reads all BMP images from each class folder, converts to grayscale,
    # resizes to 32x32, and assigns integer labels based on class index
    images = []
    labels = []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        print(f"Loading {class_name}...")
        for filename in os.listdir(class_dir):
            if filename.endswith(".bmp"):
                img_path = os.path.join(class_dir, filename)
                img = Image.open(img_path).convert("L")  # Convert to grayscale
                img = img.resize((IMG_SIZE, IMG_SIZE))   # Resize to 32x32
                images.append(np.array(img))
                labels.append(label_idx)
    return np.array(images), np.array(labels)

print("Loading dataset...")
X, y = load_dataset()

# Normalize pixel values from [0, 255] to [0, 1] for faster, more stable training
X = X.astype("float32") / 255.0

# Add channel dimension: (N, 32, 32) -> (N, 32, 32, 1) required by Conv2D
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Split 80% training / 20% test, stratified to ensure equal class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# --- CNN Architecture ---
# Compact architecture designed to fit within ESP32-S3 memory constraints.
# Three convolutional blocks progressively extract features from the image,
# followed by two fully connected layers for classification.
model = keras.Sequential([
    keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    # Block 1: detects low-level features (edges, gradients) — 32x32x16
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),        # 32x32 -> 16x16

    # Block 2: detects mid-level features (curves, finger outlines) — 16x16x32
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),        # 16x16 -> 8x8

    # Block 3: detects high-level hand shape features — 8x8x32
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),        # 8x8 -> 4x4, flattens to 512 (not 4096)

    layers.Flatten(),                   # 4x4x32 = 512 features
    layers.Dense(32, activation="relu"),# Learns feature combinations
    layers.Dropout(0.3),               # Randomly disables 30% of neurons to prevent overfitting
    layers.Dense(3, activation="softmax") # Output: probability for each of 3 classes
])

# Adam optimizer adapts learning rate automatically
# Sparse categorical crossentropy is used because labels are integers (0, 1, 2)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- Training ---
print("\nTraining...")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=(X_test, y_test))

# Evaluate final accuracy on held-out test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc:.2%}")

# Save full Keras model for weight export and future retraining
model.save(os.path.expanduser("~/Desktop/rps_model_small.keras"))
print("Keras model saved!")

# --- Convert to TFLite ---
# TFLite is a compressed format for running models on embedded devices.
# Uses concrete function approach for compatibility with tensorflow-macos on M2.
print("Converting to TFLite...")

@tf.function(input_signature=[tf.TensorSpec(shape=[1, 32, 32, 1], dtype=tf.float32)])
def predict(x):
    # Wraps model inference as a concrete TensorFlow function for TFLite export
    return model(x)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [predict.get_concrete_function()]
)
tflite_model = converter.convert()

with open(os.path.expanduser("~/Desktop/rps_model_small.tflite"), "wb") as f:
    f.write(tflite_model)
print("TFLite model saved!")

# Print final model size to confirm it fits on the ESP32-S3's 8MB flash
size = os.path.getsize(os.path.expanduser("~/Desktop/rps_model_small.tflite"))
print(f"TFLite model size: {size/1024:.1f} KB")