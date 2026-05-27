import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = tf.keras.models.load_model("model/brain_model.h5")

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Folder containing test images
sample_folder = "samples"

# Loop through images
for img_name in os.listdir(sample_folder):

    img_path = os.path.join(sample_folder, img_name)

    # Load image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    print(f"{img_name} --> {predicted_class}")