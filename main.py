import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ===============================
# Paths
# ===============================
train_path = r"E:/image_classifair/dataset/train"
test_path = r"E:/image_classifair/dataset/test"

# ===============================
# Image settings
# ===============================
img_size = (224, 224)
batch_size = 64

# ===============================
# Load Train Dataset
# ===============================
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    color_mode="rgb"   # Force 3 channels (fix 2-channel error)
)

# ===============================
# Load Test Dataset
# ===============================
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    color_mode="rgb"
)

# ✅ Save class names BEFORE ignore_errors
class_names = train_dataset.class_names
print("Classes:", class_names)

# ===============================
# Ignore Corrupted Images
# ===============================
train_dataset = train_dataset.ignore_errors()
test_dataset = test_dataset.ignore_errors()

# ===============================
# Performance Optimization
# ===============================
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# ===============================
# Load MobileNetV2 Base Model
# ===============================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # No fine tuning

# ===============================
# Build Model
# ===============================
model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# ===============================
# Compile Model
# ===============================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Training started...")

# ===============================
# Train
# ===============================
history = model.fit(
    train_dataset,
    epochs=10
)

print("Training finished!")

# ===============================
# Evaluate on Test Dataset
# ===============================
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Accuracy:", test_accuracy)

# ===============================
# Save Model
# ===============================
model.save("dog_cat_25k_final.h5")
print("Model saved successfully!")