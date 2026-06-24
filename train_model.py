import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json
import os

# -----------------------------
# PARAMETERS
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
DATASET_DIR = "dataset/train"

# -----------------------------
# CREATE MODEL FOLDER
# -----------------------------
os.makedirs("model", exist_ok=True)

# -----------------------------
# DATA GENERATOR WITH AUGMENTATION
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# -----------------------------
# SAVE CLASS INDICES
# -----------------------------
class_indices = train_generator.class_indices

with open("model/class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=4)

num_classes = len(class_indices)

print("✅ Classes:", class_indices)

# -----------------------------
# CNN MODEL
# -----------------------------
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("model/plant_model.h5")

# -----------------------------
# SAVE TRAINING HISTORY
# -----------------------------
with open("model/training_history.json", "w") as f:
    json.dump(history.history, f, indent=4)

print("\n🎉 TRAINING COMPLETED")
print("📁 Model saved → model/plant_model.h5")
print("📁 Class indices → model/class_indices.json")
print("📁 Training history → model/training_history.json")