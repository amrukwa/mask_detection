import numpy as np
import cv2 as cv
from model.preprocessing import get_faces
from model.model import prepare_result
from joblib import load
import matplotlib.pyplot as plt

face_detector = cv.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
mask_detector = load('model/mask_detector.joblib')
img = cv.imread("dataset/other_example.png")
if get_faces(img, face_detector, 1.2, 5) is not None:
    faces, coords = get_faces(img, face_detector, 1.2, 5, for_display=True)
    faces = [face for image in faces for face in image]
    faces_flattened = (np.array(faces)).reshape((len(coords), -1))
    res = mask_detector.predict(faces_flattened)
    img_display = prepare_result(img, coords, res)
    plt.imshow(img_display)

else:
    print("No face detected")
