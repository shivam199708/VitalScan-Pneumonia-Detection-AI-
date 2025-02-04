import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image

# Define dataset paths
TRAIN_DIR = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/data/train"
VAL_DIR = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/data/val"
TEST_DIR = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/data/test"
RESULTS_DIR = "results/eda_output"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Function to count images in each class
def count_images(directory):
    categories = os.listdir(directory)
    class_counts = {category: len(os.listdir(os.path.join(directory, category))) for category in categories}
    return class_counts

# Get dataset distribution
train_counts = count_images(TRAIN_DIR)
val_counts = count_images(VAL_DIR)
test_counts = count_images(TEST_DIR)

# Save class distribution to CSV
class_distribution_df = pd.DataFrame([train_counts, val_counts, test_counts], index=["Train", "Validation", "Test"])
class_distribution_df.to_csv(os.path.join(RESULTS_DIR, "class_distribution.csv"))

# Plot class distribution
def plot_class_distribution(counts, title, filename):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="Blues")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.show()

plot_class_distribution(train_counts, "Training Data Distribution", "train_distribution.png")
plot_class_distribution(val_counts, "Validation Data Distribution", "val_distribution.png")
plot_class_distribution(test_counts, "Test Data Distribution", "test_distribution.png")

# Function to display and save sample images
def show_sample_images(directory, filename, num_images=3):
    categories = os.listdir(directory)
    plt.figure(figsize=(12, 6))
    for idx, category in enumerate(categories):
        img_path = os.path.join(directory, category, os.listdir(os.path.join(directory, category))[0])
        img = Image.open(img_path)
        plt.subplot(1, len(categories), idx + 1)
        plt.imshow(img, cmap="gray")
        plt.title(category)
        plt.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.show()

print("Displaying sample images from training set...")
show_sample_images(TRAIN_DIR, "sample_images.png")

# Data augmentation visualization
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

def visualize_augmentation(directory, filename):
    category = os.listdir(directory)[0]  # Choose first category
    img_path = os.path.join(directory, category, os.listdir(os.path.join(directory, category))[0])
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    plt.figure(figsize=(10, 5))
    for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(image.array_to_img(batch[0]))
        plt.axis("off")
        if i == 4:
            break
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.show()

print("Visualizing data augmentation...")
visualize_augmentation(TRAIN_DIR, "augmented_images.png")
