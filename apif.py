from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from io import BytesIO
from flask_cors import CORS
app = Flask(__name__)
CORS(app)                                                                                                                                       
model = load_model('D:\progect\\trained_plant_disease_model.keras')

validation_set = tf.keras.utils.image_dataset_from_directory(
    'D:\progect\\valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

class_names = validation_set.class_names

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400
    
    image_file = request.files['image']
    
    image_bytes = image_file.read()
    
    image = load_img(BytesIO(image_bytes), target_size=(128, 128))
    
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    
    return jsonify({'class': predicted_class}), 200

if __name__ == '__main__':
    app.run(debug=True)
