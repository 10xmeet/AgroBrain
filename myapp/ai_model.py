import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Get absolute path to the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'myapp', 'models', 'plant_disease_model.h5')

# Ensure the model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

# Load model
model = load_model(MODEL_PATH)

# Dynamically extract class labels
CLASS_LABELS = list(model.class_names) if hasattr(model, 'class_names') else [f"Class {i}" for i in range(model.output_shape[-1])]

def predict_disease(image_path):
    """
    Function to predict plant disease using the pre-trained AI model.
    Dynamically extracts class labels.
    """
    img = load_img(image_path, target_size=(160, 160))  # Resize
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    disease_name = CLASS_LABELS[predicted_class] if predicted_class < len(CLASS_LABELS) else "Unknown Disease"

    return {
        "Plant Type": disease_name.split()[0] if " " in disease_name else "Unknown",
        "Disease Detected": disease_name,
        "Confidence Level": f"{max(prediction[0]) * 100:.2f}%",
        "Plant Health Bar": int((1 - max(prediction[0])) * 100)  # Lower confidence means worse health
    }
