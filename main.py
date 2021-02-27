import numpy as np
import cv2 as cv
from models.preprocessing import get_faces
from models.model import prepare_result
from joblib import load
from sklearn.ensemble import RandomForestClassifier

face_detector = cv.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
mask_detector = load('models/mask_detector.joblib')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    if get_faces(img, face_detector, 1.2, 3) is not None:
        faces, coords = get_faces(img, face_detector, 1.2, 3, for_display=True)
        faces = [face for image in faces for face in image]
        faces_flattened = (np.array(faces)).reshape((len(coords), -1))
        res = mask_detector.predict(faces_flattened)
        img_display = prepare_result(img, coords, res, is_matlpotlib=False)
    else:
        img_display = img

    cv.imshow('frame', img_display)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()