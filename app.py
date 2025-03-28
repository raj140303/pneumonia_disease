import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

model = load_model("model_weights/vgg_unfrozen_fixed.keras")

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_class_name(class_no):
    return "Normal" if class_no == 0 else "Pneumonia"

def preprocess_image(img_path):
    try:
        print(f"Processing image: {img_path}")
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Invalid image file")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (128, 128))  # Resize to model input size
        image = np.array(image) / 255.0  # Normalize
        input_img = np.expand_dims(image, axis=0)  # Add batch dimension
        return input_img
    
    except Exception as e:
        return str(e)

def getResult(img_path):
    try:
        input_img = preprocess_image(img_path)
        if isinstance(input_img, str):
            return input_img 
        
        # Model Prediction
        result = model.predict(input_img)
        print(f"Model Output: {result}")
        result01 = np.argmax(result, axis=1)
        return result01
    
    except Exception as e:
        return str(e)

# routes 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)
        print(f"File saved at: {file_path}")

        
        result = getResult(file_path)
        if isinstance(result, str):
            return jsonify({'error': result}), 400

        class_name = get_class_name(result[0])
        return jsonify({'prediction': class_name}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200

if __name__ == '__main__':
    app.run(debug=True)