import os
import glob
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
try:
    modeln = load_model("nail_diagnosis_vgg16.h5")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clear_upload_folder():
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for f in files:
        os.remove(f)

# Class index
class_labels = [
    'Dariers disease', 'Muehrcke’s lines', 'Alopecia areata', 'Beau’s lines', 'Bluish nail',
    'Clubbing, eczema, half-and-half nails (Lindsay’s nails)', 'Koilonychia', 'Leukonychia',
    'Onycholysis', 'Pale nail', 'Red lunula', 'Splinter hemorrhage',
    'Terry’s nail', 'White nail', 'Yellow nails'
]

# Routes
@app.route('/')
@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/nailprediction')
def nailprediction():
    return render_template('nailprediction.html')

@app.route('/nailresult', methods=["GET", "POST"])
def nres():
    if request.method == "POST":
        if 'image' not in request.files:
            return "No file part", 400

        f = request.files['image']

        if f.filename == '':
            return "No selected file", 400

        if not allowed_file(f.filename):
            return "Invalid file type. Only PNG, JPG, JPEG allowed.", 400

        filename = secure_filename(f.filename)
        clear_upload_folder()  # Clean up old files
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        f.save(filepath)

        try:
            # Load and preprocess image
            img = image.load_img(filepath, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            img_data = preprocess_input(x)

            # Sanity check: modeln.predict must be callable
            if not callable(modeln.predict):
                return "Prediction failed: modeln.predict is not callable.", 500

            # Run prediction
            predictions = modeln.predict(img_data)

            if not isinstance(predictions, (np.ndarray, list)):
                return "Prediction failed: Model did not return an array or list.", 500

            prediction_array = predictions[0]
            prediction_index = int(np.argmax(prediction_array))
            nresult = class_labels[prediction_index]
            confidence = round(float(np.max(prediction_array)) * 100, 2)

            return render_template(
                'nailresult.html',
                nresult=nresult,
                confidence=confidence,
                filename=filename
            )

        except Exception as e:
            import traceback
            return f"<pre>Prediction failed: {e}\n\n{traceback.format_exc()}</pre>", 500

    return redirect(url_for('nailprediction'))

# Run app
if __name__ == "__main__":
    app.run(debug=True, port=8080)
