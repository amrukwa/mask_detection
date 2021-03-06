import cv2 as cv
import numpy as np

mask_result = ["No mask!", "Incorrect!", "Mask"]
font = cv.FONT_HERSHEY_SIMPLEX

def prepare_result(img, coords, results, is_matlpotlib=True):
    if is_matlpotlib:
        img_disp = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask_coloring = [(255, 0, 0), (255, 0, 0), (0, 255, 0)]
    else:
        img_disp = img
        mask_coloring = [(0, 0, 255), (0, 0, 255), (0, 255, 0)]
    i = 0
    for coord in coords:
        cv.rectangle(img_disp, (coord[0], coord[1]), (coord[2], coord[3]), mask_coloring[results[i]], 2)
        cv.putText(img_disp, mask_result[results[i]],(coord[0], coord[1]-10), fontFace = font, fontScale=0.4, color=mask_coloring[results[i]],thickness=1, lineType=cv.LINE_AA)
        i += 1
    return img_disp


if __name__ == "__main__":
    from preprocessing import get_faces
    from joblib import dump
    import glob
    from sklearn.ensemble import RandomForestClassifier
    
    exts = ["jpg","png"]
    folders = ["dataset/without_mask/", "dataset/incorrect_mask/", "dataset/with_mask/"]
    face_detector = cv.dnn.readNetFromCaffe("dataset/deploy.prototxt.txt", "dataset/res10_300x300_ssd_iter_140000.caffemodel")
    data = []
    labels = []
    for i in range(len(folders)):
        from_folder = [glob.glob(folders[i]+'*.%s' % ext) for ext in exts]
        images = [cv.imread(img) for i in from_folder for img in i if cv.imread(img) is not None]
        faces = [get_faces(image, face_detector) for image in images if get_faces(image, face_detector) is not None]
        faces = [face for image in faces for face in image]
        labeled = [i]*len(faces)
        data = data + faces
        labels = labels + labeled
    Data = (np.array(data)).reshape((len(labels), -1))
    mask_detector = RandomForestClassifier().fit(Data, labels)
    dump(mask_detector, 'models/mask_detector.joblib') 
