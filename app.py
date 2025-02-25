import os
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Define model path and Google Drive file ID
MODEL_PATH = "models/best_crop_disease_model.keras"
GDRIVE_FILE_ID = "1ovzbUpvFsrA04z5WgyOxlbqxGmnKQ_nm"  # Your Google Drive model file ID

def download_model():
    """Downloads model from Google Drive if not present locally."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        os.makedirs("models", exist_ok=True)
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

# Ensure the model is downloaded before loading
download_model()

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Set model to None if it fails to load

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

# Prevention measures
prevention_measures = {
    "Apple___Apple_scab": "Prune infected branches, apply fungicides, and plant resistant varieties.",
    "Apple___Black_rot": "Remove infected fruit, use copper-based fungicides, and ensure proper air circulation.",
    "Apple___Cedar_apple_rust": "Remove nearby juniper plants, apply fungicides, and plant resistant apple varieties.",
    "Apple___healthy": "No disease detected! Maintain proper watering, fertilization, and regular inspection.",
    "Blueberry___healthy": "No disease detected! Ensure good soil drainage and regular monitoring.",
    "Cherry_(including_sour)___Powdery_mildew": "Prune infected branches, improve air circulation, and apply sulfur-based fungicides.",
    "Corn_(maize)___Common_rust_": "Plant resistant varieties, apply fungicides early, and remove infected leaves.",
    "Potato___Early_blight": "Apply fungicides, rotate crops, and remove infected leaves.",
    "Tomato___Bacterial_spot": "Apply copper-based sprays, avoid overhead watering, and remove infected plants.",
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

    img = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert to RGB
    img = img.resize((128, 128))  # Resize image
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand for model input

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Get disease name and prevention measures
    disease_name = class_labels.get(predicted_class, "Unknown Disease")
    prevention = prevention_measures.get(disease_name, "Remove infected leaves, apply fungicides, and improve air circulation.")

    return {"disease": disease_name, "prevention": prevention}

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

    # Read file and make prediction
    image_data = file.read()
    result = predict_disease(image_data)

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

