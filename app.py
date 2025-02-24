import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained model
model = load_model("models/best_crop_disease_model.keras")

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
    "Potato___Early_blight": "Apply fungicides, rotate crops, and remove infected leaves.",
    "Potato___Late_blight": "Remove and destroy infected plants, use fungicides, and avoid wet foliage.",
    "Potato___healthy": "No disease detected! Ensure well-drained soil and regular inspection.",
    "Tomato___Bacterial_spot": "Apply copper-based sprays, avoid overhead watering, and remove infected plants.",
    "Tomato___Early_blight": "Use disease-resistant varieties, apply fungicides, and rotate crops.",
    "Tomato___Late_blight": "Remove infected plants, apply fungicides, and ensure proper spacing between plants.",
    "Tomato___healthy": "No disease detected! Ensure proper watering, fertilization, and pest control."
}

# Function to predict disease and provide prevention measures
def predict_disease(image_data):
    img = Image.open(io.BytesIO(image_data))
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
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read file and make prediction
    image_data = file.read()
    result = predict_disease(image_data)

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)