"""MNIST Hand-Written Digit Classification with TensorFlow/Keras.

Author: Arunangshu Debnath, DESY Hamburg
Updated for modern TensorFlow 2.x standards.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Suppress unnecessary TensorFlow logs
tf.get_logger().setLevel(logging.ERROR)
print(f"TensorFlow Version: {tf.__version__}")

# -------------------------------------------------------------------------
# 1. Dataset Loading and Inspection
# -------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"x_train shape: {x_train.shape} | y_train shape: {y_train.shape}")
print(f"x_test shape:  {x_test.shape}  | y_test shape:  {y_test.shape}")
print(f"Unique classes: {np.unique(y_train)}")

# Inspect a single training sample
plt.figure(figsize=(3, 3))
plt.imshow(x_train[0], cmap="binary")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()

# -------------------------------------------------------------------------
# 2. Data Preprocessing
# -------------------------------------------------------------------------
# Reshape (N, 28, 28) -> (N, 784) and scale pixel intensities to [0, 1]
x_train_flat = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test_flat = x_test.reshape(-1, 784).astype("float32") / 255.0

# One-hot encode targets: (N,) -> (N, 10)
y_train_encoded = keras.utils.to_categorical(y_train, num_classes=10)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes=10)

print(f"x_train_flat shape: {x_train_flat.shape}")
print(f"y_train_encoded shape: {y_train_encoded.shape}")

# -------------------------------------------------------------------------
# 3. Model Architecture
# -------------------------------------------------------------------------
model = keras.Sequential(
    [
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ],
    name="mnist_dense_classifier",
)

model.compile(
    optimizer="adam",  # Adam generally converges faster and more reliably than vanilla SGD
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -------------------------------------------------------------------------
# 4. Model Training & Evaluation
# -------------------------------------------------------------------------
history = model.fit(
    x_train_flat,
    y_train_encoded,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
)

test_loss, test_acc = model.evaluate(x_test_flat, y_test_encoded, verbose=0)
print(f"\nTest Set Loss:     {test_loss:.4f}")
print(f"Test Set Accuracy: {test_acc * 100:.2f}%")

# -------------------------------------------------------------------------
# 5. Predictions & Visualization
# -------------------------------------------------------------------------
preds = model.predict(x_test_flat)

# Plot a 5x5 grid of sample predictions
plt.figure(figsize=(10, 10))
start_index = 0

for i in range(25):
    idx = start_index + i
    pred_label = np.argmax(preds[idx])
    true_label = y_test[idx]

    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # Highlight correct predictions in green, errors in red
    color = "forestgreen" if pred_label == true_label else "crimson"

    plt.imshow(x_test[idx], cmap="binary")
    plt.xlabel(f"Pred: {pred_label} (True: {true_label})", color=color, fontsize=9)

plt.tight_layout()
plt.show()

# Softmax confidence distribution for a single sample
sample_id = 8
plt.figure(figsize=(6, 3))
plt.bar(range(10), preds[sample_id], color="steelblue")
plt.xticks(range(10))
plt.xlabel("Digit Class")
plt.ylabel("Softmax Probability")
plt.title(f"Class Distribution for Sample {sample_id} (True: {y_test[sample_id]})")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
