import numpy as np
import cv2 as cv
from models.preprocessing import get_faces
from models.model import prepare_result
from joblib import load
import matplotlib.pyplot as plt
import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier

face_detector = cv.dnn.readNetFromCaffe("dataset/deploy.prototxt.txt", "dataset/res10_300x300_ssd_iter_140000.caffemodel")
mask_detector = load('models/mask_detector.joblib')

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
camera = cv.VideoCapture(0)

def gen_frames():  
    while True:
        success, img = camera.read()
        if not success:
            break
        else:
            if get_faces(img, face_detector) is not None:
                faces, coords = get_faces(img, face_detector, for_display=True)
                faces_flattened = (np.array(faces)).reshape((len(coords), -1))
                res = mask_detector.predict(faces_flattened)
                img = prepare_result(img, coords, res, is_matlpotlib=False)
            ret, buffer = cv.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def choose_method():
    if request.method == 'GET':
        return render_template('choose_app.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    img = cv.imread("uploads/" + filename)
    if get_faces(img, face_detector) is not None:
        faces, coords = get_faces(img, face_detector, for_display=True)
        faces_flattened = (np.array(faces)).reshape((len(coords), -1))
        res = mask_detector.predict(faces_flattened)
        img_display = prepare_result(img, coords, res, is_matlpotlib=False)
        cv.imwrite("uploads/display.png", img_display)
        file_name = 'display.png'
    else:
        file_name = filename
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               file_name)

@app.route('/camera')
def access_camera():
    return render_template('stream.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')