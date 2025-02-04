import gradio as gr
import tensorflow.lite as tflite
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image

# Define model path
MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/VitalScan-Pneumonia-Detection-AI-/models/pneumonia_model.tflite"

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img):
    """Preprocess the uploaded image for TFLite model prediction."""
    img = img.resize((150, 150))  # Resize to match model input shape
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def generate_image_statistics(img):
    """Generate image statistics and histogram."""
    img_gray = img.convert('L')  # Convert to grayscale
    img_array = np.array(img_gray)
    
    # Generate histogram
    plt.figure(figsize=(5, 3))
    plt.hist(img_array.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("histogram.png")
    
    return "histogram.png"

def predict_pneumonia(img):
    """Run inference using the TFLite model and provide recommendations."""
    img_array = preprocess_image(img)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Extract result
    score = prediction[0][0]
    result = "Pneumonia" if score > 0.5 else "Normal"
    
    # Generate recommendations based on the result
    recommendations = """If the result is Pneumonia, consult a doctor immediately.
    Stay hydrated, rest, and follow prescribed medication. 
    Maintain a healthy diet and avoid exposure to infections."""
    
    # Generate image statistics
    hist_path = generate_image_statistics(img)
    
    return f"🩺 Prediction: {result} (Confidence Score: {score:.4f})\n\n📌 Recommendations:\n{recommendations}", hist_path

# Create Gradio interface
gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(), gr.Image(type="filepath")],
    title="🩺 Pneumonia Detection AI",
    description="Upload a chest X-ray image to check for pneumonia. The model will analyze the image and provide a confidence score, along with recommendations if needed. The app also generates a pixel intensity histogram of the uploaded image.",
    live=True
).launch(share=True)
