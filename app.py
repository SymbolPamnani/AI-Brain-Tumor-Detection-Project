from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/brain_model.h5")

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Home route
@app.route('/')
def home():
    return "Brain Tumor Detection API is Running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Check if image exists
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Read image
    img = Image.open(io.BytesIO(file.read())).convert('RGB')

    # Resize image
    img = img.resize((224, 224))

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = float(np.max(prediction))

    # Return result
    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence * 100, 2)
    })

# Run server
if __name__ == '__main__':
    app.run(debug=True)