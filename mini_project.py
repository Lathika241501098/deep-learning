
# ============================================================
# DEEP LEARNING MINI PROJECT
# Fashion Item Image Classification:
# Baseline Neural Network vs CNN
# Dataset: Fashion-MNIST
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# STEP 1: INITIALIZATION
# ------------------------------------------------------------

SEED = 42
EPOCHS = 10
BATCH_SIZE = 128

np.random.seed(SEED)
tf.random.set_seed(SEED)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("TensorFlow version:", tf.__version__)

# ------------------------------------------------------------
# STEP 2: LOAD DATASET
# Downloads automatically through Keras on first use
# ------------------------------------------------------------

(x_train_full, y_train_full), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

print("\nDataset loaded successfully!")
print("Training images:", x_train_full.shape)
print("Training labels:", y_train_full.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

# Display sample images
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train_full[i], cmap="gray")
    plt.title(class_names[y_train_full[i]])
    plt.axis("off")

plt.suptitle("Sample Fashion-MNIST Images")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# STEP 3: PREPROCESS DATA
# ------------------------------------------------------------

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension for CNN
x_train_full = x_train_full[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("\nAfter preprocessing:")
print("Training data:", x_train_full.shape)
print("Test data:", x_test.shape)
print("Pixel range:", x_train_full.min(), "to", x_train_full.max())

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train_full
)

print("\nTraining set:", x_train.shape)
print("Validation set:", x_val.shape)
print("Test set:", x_test.shape)

# ------------------------------------------------------------
# STEP 4: BUILD BASELINE NEURAL NETWORK (MLP)
# ------------------------------------------------------------

def build_baseline():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ], name="baseline_mlp")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


baseline_model = build_baseline()

print("\nBASELINE MODEL SUMMARY")
baseline_model.summary()

# ------------------------------------------------------------
# STEP 5: TRAIN BASELINE MODEL
# ------------------------------------------------------------

print("\nTraining baseline model...")

hist_base = baseline_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)

# ------------------------------------------------------------
# STEP 6: BUILD CNN MODEL
# ------------------------------------------------------------

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64, (3, 3), activation="relu", padding="same"
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax")
    ], name="improved_cnn")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


cnn_model = build_cnn()

print("\nCNN MODEL SUMMARY")
cnn_model.summary()

# ------------------------------------------------------------
# STEP 7: TRAIN CNN
# ------------------------------------------------------------

print("\nTraining CNN model...")

hist_cnn = cnn_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)

# ------------------------------------------------------------
# STEP 8: TRAINING ACCURACY AND LOSS GRAPHS
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(hist_base.history["accuracy"], label="Baseline train")
axes[0].plot(hist_base.history["val_accuracy"], label="Baseline validation")
axes[0].plot(hist_cnn.history["accuracy"], label="CNN train")
axes[0].plot(hist_cnn.history["val_accuracy"], label="CNN validation")
axes[0].set_title("Training and Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(hist_base.history["loss"], label="Baseline train")
axes[1].plot(hist_base.history["val_loss"], label="Baseline validation")
axes[1].plot(hist_cnn.history["loss"], label="CNN train")
axes[1].plot(hist_cnn.history["val_loss"], label="CNN validation")
axes[1].set_title("Training and Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_performance.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# STEP 9: EVALUATE MODELS ON TEST DATA
# ------------------------------------------------------------

def evaluate(model, name):
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1-score": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )
    }

    print(f"\n{'=' * 15} {name} {'=' * 15}")

    for metric, score in results.items():
        print(f"{metric}: {score:.4f} ({score * 100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
        digits=4
    ))

    return results, confusion_matrix(y_test, y_pred), y_pred


base_results, base_cm, base_pred = evaluate(
    baseline_model, "Baseline MLP"
)

cnn_results, cnn_cm, cnn_pred = evaluate(
    cnn_model, "Convolutional Neural Network"
)

# ------------------------------------------------------------
# STEP 10: CONFUSION MATRICES
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(
    base_cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[0]
)

axes[0].set_title("Baseline MLP Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(
    cnn_cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axes[1]
)

axes[1].set_title("CNN Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# STEP 11: COMPARE BOTH MODELS
# ------------------------------------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

print("\n" + "=" * 60)
print("MODEL COMPARISON ON TEST DATA")
print("=" * 60)
print(f"{'Metric':<15}{'Baseline':>15}{'CNN':>15}{'Difference':>15}")

for metric in metrics:
    base = base_results[metric]
    cnn = cnn_results[metric]

    print(
        f"{metric:<15}{base:>15.4f}"
        f"{cnn:>15.4f}{cnn - base:>15.4f}"
    )

print("\nTrainable and non-trainable parameter totals:")
print("Baseline parameters:", baseline_model.count_params())
print("CNN parameters:", cnn_model.count_params())

# Comparison graph
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(
    x - width / 2,
    [base_results[m] for m in metrics],
    width,
    label="Baseline MLP"
)
plt.bar(
    x + width / 2,
    [cnn_results[m] for m in metrics],
    width,
    label="CNN"
)

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Baseline MLP vs CNN")
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# STEP 12: SAMPLE IMAGE PREDICTION
# ------------------------------------------------------------

i = 0
image = x_test[i:i + 1]

base_probs = baseline_model.predict(image, verbose=0)[0]
cnn_probs = cnn_model.predict(image, verbose=0)[0]

base_label = np.argmax(base_probs)
cnn_label = np.argmax(cnn_probs)

plt.figure(figsize=(5, 4))
plt.imshow(x_test[i].squeeze(), cmap="gray")
plt.title(
    f"Actual: {class_names[y_test[i]]}\n"
    f"Baseline: {class_names[base_label]}\n"
    f"CNN: {class_names[cnn_label]}"
)
plt.axis("off")
plt.savefig("sample_prediction.png", dpi=200)
plt.show()

print("\nSAMPLE PREDICTION")
print("Actual class:", class_names[y_test[i]])
print("Baseline prediction:", class_names[base_label])
print("CNN prediction:", class_names[cnn_label])
print(f"Baseline confidence: {base_probs[base_label]:.2%}")
print(f"CNN confidence: {cnn_probs[cnn_label]:.2%}")

# ------------------------------------------------------------
# STEP 13: SAVE TRAINED MODELS
# ------------------------------------------------------------

baseline_model.save("baseline_model.keras")
cnn_model.save("fashion_cnn_model.keras")

print("\nModels saved successfully!")

# ------------------------------------------------------------
# STEP 14: CONCLUSION BASED ON ACTUAL ACCURACY
# ------------------------------------------------------------

print("\nPROJECT CONCLUSION")

if cnn_results["Accuracy"] > base_results["Accuracy"]:
    print("CNN achieved higher test accuracy than the baseline MLP.")
elif cnn_results["Accuracy"] < base_results["Accuracy"]:
    print("Baseline MLP achieved higher test accuracy than the CNN.")
else:
    print("Both models achieved equal test accuracy.")

print("Baseline test accuracy:", f"{base_results['Accuracy']:.2%}")
print("CNN test accuracy:", f"{cnn_results['Accuracy']:.2%}")

print("\nGenerated output files:")
print("- sample_images.png")
print("- training_performance.png")
print("- confusion_matrices.png")
print("- model_comparison.png")
print("- sample_prediction.png")
print("- baseline_model.keras")
print("- fashion_cnn_model.keras")
