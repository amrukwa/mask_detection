import cv2 as cv
import numpy as np

def get_faces(img, model, dimensions=(240, 240), for_display=False):
    h, w = img.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    model.setInput(blob)
    detected = model.forward()
    extracted = []
    coordinates = []
    for i in range(detected.shape[2]):
        confidence = detected[0, 0, i, 2]
        if confidence > 0.5:
            box = detected[0, 0, i, 3:7] * np.array([w, h, w, h])
            coords = box.astype("int")
            crop_img = img[coords[1]:coords[3], coords[0]:coords[2]]
            crop_img = cv.resize(crop_img, dimensions)
            crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
            extracted.append(crop_img)
            if for_display:
                coordinates.append(coords)
    if len(extracted)==0:
        return None
    if for_display:
        return extracted, coordinates
    return extracted
