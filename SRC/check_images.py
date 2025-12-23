import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- 1. Path Configuration ---
model_path = os.path.join("..", "Models", "cnn_mask_detector_v2.h5")
# Replace with the actual path of the image you want to test
test_image_path = os.path.join("..", "path_to_your_example_image.png") 

# --- 2. Load the Trained Model ---
if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
else:
    print(f"❌ Error: Model not found at {os.path.abspath(model_path)}")
    exit()

def predict_mask(img_path):
    """Loads an image and predicts if a mask is being worn."""
    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at {os.path.abspath(img_path)}")
        return

    # 3. Image Preprocessing
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create batch dimension
    img_array /= 255.0 # Normalization

    # 4. Prediction Logic
    prediction = model.predict(img_array)
    
    # Class logic based on alphabetical folder order (With_Mask vs Without_Mask)
    if prediction[0][0] < 0.5:
        result = "Wearing Mask"
        confidence = (1 - prediction[0][0]) * 100
    else:
        result = "No Mask Detected"
        confidence = prediction[0][0] * 100

    print(f"Image File: {os.path.basename(img_path)}")
    print(f"Result: {result}")
    print(f"Confidence Level: {confidence:.2f}%")
    print("-" * 30)

# Run Prediction
predict_mask(test_image_path)
