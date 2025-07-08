import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/efficientnet_nn_model.h5'  
IMG_SIZE = 224

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained Keras model
model = load_model(MODEL_PATH)

# Your class labels
class_names = ['American Football', 'Basketball', 'Soccer', 'Tennis', 'Volleyball']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', result="No image part in request")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', result="No selected image")

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load & preprocess image
            img = load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            predicted_class = class_names[np.argmax(preds)]

            return render_template('index.html', image=filename, result=predicted_class)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
