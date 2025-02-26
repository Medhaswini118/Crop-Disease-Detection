import os
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io



app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow all origins for /predict
# Allow cross-origin requests

# Define model path and Google Drive file ID
MODEL_PATH = "models/best_crop_disease_model.keras"
GDRIVE_FILE_ID = "1jEvdm0UbVj6hIHWCiU7Bgm_aaleKvmN8"  # Your Google Drive model file ID

def download_model():
    """Downloads model from Google Drive if not present locally."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        os.makedirs("models", exist_ok=True)
        gdown.download("1jEvdm0UbVj6hIHWCiU7Bgm_aaleKvmN8", MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

# Ensure the model is downloaded before loading
download_model()

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Set model to None if loading fails

# Disease class labels
class_labels = {
    0: "Apple___Apple_scab", 1: "Apple___Black_rot", 2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy", 4: "Blueberry___healthy", 5: "Cherry_(including_sour)___Powdery_mildew",
    6: "Cherry_(including_sour)___healthy", 7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    8: "Corn_(maize)___Common_rust_", 9: "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___healthy", 11: "Grape___Black_rot", 12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)", 16: "Peach___Bacterial_spot",
    17: "Peach___healthy", 18: "Pepper,_bell___Bacterial_spot", 19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight", 21: "Potato___Late_blight", 22: "Potato___healthy",
    23: "Raspberry___healthy", 24: "Soybean___healthy", 25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch", 27: "Strawberry___healthy", 28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight", 30: "Tomato___Late_blight", 31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot", 33: "Tomato___Spider_mites Two-spotted_spider_mite",
    34: "Tomato___Target_Spot", 35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus", 37: "Tomato___healthy"
}

# Prevention measures (Only showing a few, you can extend it)
prevention_measures = {
    "Apple___Apple_scab": "Prune infected branches, apply fungicides, and plant resistant varieties.",
    "Apple___Black_rot": "Remove infected fruit, use copper-based fungicides, and ensure proper air circulation.",
    "Apple___Cedar_apple_rust": "Remove nearby juniper plants, apply fungicides, and plant resistant apple varieties.",
    "Apple___healthy": "No disease detected! Maintain proper watering, fertilization, and regular inspection.",
    "Blueberry___healthy": "No disease detected! Ensure good soil drainage and regular monitoring.",
    "Cherry_(including_sour)___Powdery_mildew": "Prune infected branches, improve air circulation, and apply sulfur-based fungicides.",
    "Cherry_(including_sour)___healthy": "No disease detected! Keep trees well-spaced and prune regularly.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, apply fungicides, and avoid overhead irrigation.",
    "Corn_(maize)___Common_rust_": "Plant resistant varieties, apply fungicides early, and remove infected leaves.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use disease-free seeds, rotate crops, and apply fungicides as needed.",
    "Corn_(maize)___healthy": "No disease detected! Ensure proper fertilization and pest control.",
    "Grape___Black_rot": "Remove and destroy infected leaves, apply fungicides, and prune vines for better airflow.",
    "Grape___Esca_(Black_Measles)": "Remove infected vines, avoid mechanical injuries, and apply fungicides.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Ensure proper drainage, apply fungicides, and remove infected leaves.",
    "Grape___healthy": "No disease detected! Maintain proper vineyard hygiene and pruning practices.",
    "Orange___Haunglongbing_(Citrus_greening)": "Control psyllid insects, remove infected trees, and plant disease-resistant varieties.",
    "Peach___Bacterial_spot": "Apply copper-based sprays, prune infected branches, and avoid overhead watering.",
    "Peach___healthy": "No disease detected! Maintain good soil health and monitor for pests.",
    "Pepper,_bell___Bacterial_spot": "Use copper-based bactericides, avoid overhead watering, and remove infected leaves.",
    "Pepper,_bell___healthy": "No disease detected! Rotate crops and maintain proper irrigation.",
    "Potato___Early_blight": "Apply fungicides, rotate crops, and remove infected leaves.",
    "Potato___Late_blight": "Remove and destroy infected plants, use fungicides, and avoid wet foliage.",
    "Potato___healthy": "No disease detected! Ensure well-drained soil and regular inspection.",
    "Raspberry___healthy": "No disease detected! Maintain proper pruning and irrigation practices.",
    "Soybean___healthy": "No disease detected! Use crop rotation and maintain soil fertility.",
    "Squash___Powdery_mildew": "Use sulfur-based fungicides, improve air circulation, and remove infected leaves.",
    "Strawberry___Leaf_scorch": "Remove infected leaves, improve air circulation, and apply appropriate fungicides.",
    "Strawberry___healthy": "No disease detected! Use mulch to prevent soil-borne diseases.",
    "Tomato___Bacterial_spot": "Apply copper-based sprays, avoid overhead watering, and remove infected plants.",
    "Tomato___Early_blight": "Use disease-resistant varieties, apply fungicides, and rotate crops.",
    "Tomato___Late_blight": "Remove infected plants, apply fungicides, and ensure proper spacing between plants.",
    "Tomato___Leaf_Mold": "Improve air circulation, apply fungicides, and avoid excessive moisture.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves, apply fungicides, and avoid overhead watering.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use insecticidal soap, increase humidity, and introduce natural predators.",
    "Tomato___Target_Spot": "Apply fungicides, remove infected leaves, and ensure good air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies, use resistant varieties, and remove infected plants.",
    "Tomato___Tomato_mosaic_virus": "Avoid tobacco use near plants, disinfect tools, and remove infected plants.",
    "Tomato___healthy": "No disease detected! Ensure proper watering, fertilization, and pest control."


}

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict disease and provide prevention measures
def predict_disease(image_data):
    global model
    if model is None:
        return {"error": "Model is not loaded. Please check server logs."}

    try:
        # Process image
        img = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert to RGB
        img = img.resize((128, 128))  # Resize image
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Expand for model input

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        confidence_score = round(float(np.max(predictions)) * 100, 2)  # Convert to percentage

        # Get disease name and prevention measures
        disease_name = class_labels.get(predicted_class_index, "Unknown Disease")
        prevention = prevention_measures.get(disease_name, "No specific prevention measure available.")

        return {
            "disease": disease_name,
            "confidence": confidence_score,
            "prevention": prevention
        }
    
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

# API endpoint for prediction
@app.route("/")
def home():
    return "Welcome to the Crop Disease Detection API! Use the /predict endpoint to classify plant diseases."

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a PNG or JPG image."}), 400

    try:
        # Read file and make prediction
        image_data = file.read()
        result = predict_disease(image_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 (common for Flask)
  # Default to 10000 if PORT not set
    app.run(host="0.0.0.0", port=port, debug=True)
