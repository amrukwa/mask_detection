import numpy as np
import cv2 as cv
from models.preprocessing import get_faces
from models.model import prepare_result
from joblib import load
import matplotlib.pyplot as plt
from flask import Flask, request, Response, render_template

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
# face_detector = cv.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
# mask_detector = load('models/mask_detector.joblib')
# img = cv.imread("dataset/other_example.png")
# if get_faces(img, face_detector, 1.2, 5) is not None:
#     faces, coords = get_faces(img, face_detector, 1.2, 5, for_display=True)
#     faces = [face for image in faces for face in image]
#     faces_flattened = (np.array(faces)).reshape((len(coords), -1))
#     res = mask_detector.predict(faces_flattened)
#     img_display = prepare_result(img, coords, res)
#     plt.imshow(img_display)

# else:
#     print("No face detected")
