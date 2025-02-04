import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# Define paths
MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/models/pneumonia_model.h5"
TRAIN_DIR = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/data/train"
VAL_DIR = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/data/val"

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(150, 150), batch_size=32, class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(150, 150), batch_size=32, class_mode="binary"
)

# Ensure TensorFlow version compatibility
print(f"✅ TensorFlow Version: {tf.__version__}")

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Print model summary before training
model.summary()

# Compile Model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=3)

# Save Model to Google Drive
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
