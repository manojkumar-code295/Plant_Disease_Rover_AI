import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42

MODEL_NAME = "chili_disease_model.keras"
CLASS_FILE = "class_names.txt"

# ============================================================
# GPU / CPU
# ============================================================

print("\n======================================")
print("CHILI LEAF DISEASE AI TRAINING")
print("======================================")

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(gpu)
else:
    print("\nNo GPU detected.")
    print("Training will use CPU.")

# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"\nDataset folder not found: {DATASET_DIR}"
    )

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training images...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation images...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# GET CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\n======================================")
print("CLASSES DETECTED")
print("======================================")

for i, class_name in enumerate(class_names):
    print(f"{i} -> {class_name}")

NUM_CLASSES = len(class_names)

if NUM_CLASSES != 4:
    raise ValueError(
        f"\nExpected 4 classes, but found {NUM_CLASSES}."
    )

# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(CLASS_FILE, "w") as file:
    for class_name in class_names:
        file.write(class_name + "\n")

print("\nClass names saved to:", CLASS_FILE)

# ============================================================
# DATA PIPELINE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

# ============================================================
# LOAD MOBILE NET V2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(
    shape=(224, 224, 3),
    name="input_image"
)

# Augmentation
x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Feature extraction
x = base_model(
    x,
    training=False
)

# Convert feature maps to vector
x = layers.GlobalAveragePooling2D()(x)

# Prevent overfitting
x = layers.Dropout(0.30)(x)

# Classification layer
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="disease_prediction"
)(x)

# Final model
model = keras.Model(
    inputs=inputs,
    outputs=outputs
)

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# SHOW MODEL
# ============================================================

print("\n======================================")
print("MODEL CREATED")
print("======================================")

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

# Save the model whenever validation accuracy improves
model_checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_NAME,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

# Stop if validation accuracy stops improving
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate if learning gets stuck
reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001,
    verbose=1
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n======================================")
print("TRAINING STARTED")
print("======================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint,
        early_stopping,
        reduce_learning_rate
    ]
)

# ============================================================

# ============================================================
# STAGE 2 - FINE TUNING MOBILENETV2
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2: FINE-TUNING MOBILENETV2")
print("=" * 60)

# Unfreeze MobileNetV2
base_model.trainable = True

# Keep only the last 30 layers trainable
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stable fine-tuning
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Recompile with a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

FINE_TUNE_EPOCHS = 8

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        model_checkpoint,
        early_stopping,
        reduce_learning_rate
    ]
)

# LOAD BEST MODEL
# ============================================================

print("\n======================================")
print("LOADING BEST MODEL")
print("======================================")

model = keras.models.load_model(
    MODEL_NAME
)

# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n======================================")
print("FINAL VALIDATION")
print("======================================")

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print("\nValidation Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nValidation Loss:")
print(f"{loss:.4f}")

# ============================================================
# BEST ACCURACY FROM TRAINING
# ============================================================

best_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    history.history["val_accuracy"].index(
        best_accuracy
    ) + 1
)

print("\n======================================")
print("BEST TRAINING RESULT")
print("======================================")

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Best Epoch: {best_epoch}"
)

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Chili Leaf Disease Classification Accuracy"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "accuracy_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Chili Leaf Disease Classification Loss"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "loss_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FINISHED
# ============================================================

print("\n======================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("======================================")

print("\nGenerated files:")

print("1.", MODEL_NAME)
print("2.", CLASS_FILE)
print("3. accuracy_graph.png")
print("4. loss_graph.png")

print("\nYour trained model is ready for prediction.")