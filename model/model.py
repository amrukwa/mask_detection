from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import cv2 as cv
import glob
from preprocessing import get_faces
import numpy as np

mask_result = ["No mask detected!", "Incorrect mask placement!", "Mask well worn"]
mask_coloring = [(255, 0, 0), (255, 0, 0), (0, 255, 0)]
font = cv.FONT_HERSHEY_SIMPLEX

def prepare_result(img, coords, results):
    img_disp = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    i = 0
    for coord in coords:
        cv.rectangle(img_disp, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), mask_coloring[results[i]], 2)
        cv.putText(img_disp, mask_result[results[i]],(coord[0], coord[1]-10), fontFace = font, fontScale=0.5, color=mask_coloring[results[i]],thickness=1, lineType=cv.LINE_AA)
        i += 1
    return img_disp


if __name__ == "__main__":
    folders = ["dataset/without_mask/*.jpg", "dataset/incorrect_mask/*.jpg", "dataset/with_mask/*.jpg"]
    face_detector = cv.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
    data = []
    labels = []
    for i in range(len(folders)):
        from_folder = glob.glob(folders[i])
        images = [cv.imread(img) for img in from_folder if cv.imread(img) is not None]
        faces = [get_faces(image, face_detector, 1.2, 5) for image in images if get_faces(image, face_detector, 1.2, 5) is not None]
        faces = [face for image in faces for face in image]
        labeled = [i]*len(faces)
        data = data + faces
        labels = labels + labeled
    Data = (np.array(data)).reshape((len(labels), -1))
    mask_detector = RandomForestClassifier().fit(Data, labels)
    dump(mask_detector, 'model/mask_detector.joblib') 
