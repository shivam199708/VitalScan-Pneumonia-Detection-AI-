import tensorflow as tf
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing import image

# Define paths
MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/models/pneumonia_model.h5"
TEST_DIR = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/data/test"
RESULTS_DIR = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/results"
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.json")
CONF_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
CLASS_REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")
ROC_CURVE_PATH = os.path.join(RESULTS_DIR, "roc_curve.png")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load trained model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
else:
    print(f"❌ Error: Model not found at {MODEL_PATH}")
    sys.exit(1)

def preprocess_image(img_path):
    """Load and preprocess image for prediction."""
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def evaluate_model(folder_path):
    """Evaluate model with test data and generate scores."""
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        folder_path, target_size=(150, 150), batch_size=32, class_mode="binary", shuffle=False
    )
    
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)
    y_pred = (y_pred_probs > 0.5).astype("int32")
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(CONF_MATRIX_PATH)
    plt.show()
    
    # Compute classification report
    report = classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"], output_dict=True)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))
    
    # Save classification report
    with open(CLASS_REPORT_PATH, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))
    
    # Compute ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(ROC_CURVE_PATH)
    plt.show()
    
    print(f"✅ Confusion matrix saved at: {CONF_MATRIX_PATH}")
    print(f"✅ Classification report saved at: {CLASS_REPORT_PATH}")
    print(f"✅ ROC curve saved at: {ROC_CURVE_PATH}")

if __name__ == "__main__":
    evaluate_model(TEST_DIR)
