from flask import Flask, render_template, request, flash, redirect, send_from_directory
import numpy as np
import cv2
import tensorflow as tf
import os
import urllib.request
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)
model = tf.keras.models.load_model('mymodel_cnn.hdf5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def man():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def home():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file uploaded')
            return render_template('home.html')
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return render_template('home.html')

        if file and allowed_file(file.filename):
            base = os.path.dirname(__file__)
            file_path = os.path.join(base, 'uploads', secure_filename(file.filename))
            file.save(file_path)

            image = cv2.imread(file_path, 0)
            image = cv2.resize(image, (28, 28))
            prediction = model.predict(image)
            pred = np.argmax(prediction, axis=1)
            return render_template('next.html', data=pred)
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
