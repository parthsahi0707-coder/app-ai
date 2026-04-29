from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Model load kar rahe hain
# Ensure "insect_model.keras" root folder mein ho
model = tf.keras.models.load_model('insect_model.keras')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((160, 160)) # Notebook wala size
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    # 0 = ants, 1 = bees
    class_names = ['ants', 'bees']
    result = class_names[np.argmax(prediction)]
    
    return jsonify({'prediction': result})

# Vercel needs this
def handler(event, context):
    return app(event, context)
