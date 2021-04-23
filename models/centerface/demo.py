import cv2
import scipy.io as sio
import os
from centerface import CenterFace
import sys

def test_image(filename):
    frame = cv2.imread(filename)
    h, w = frame.shape[:2]
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
    else:
        dets = centerface(frame, threshold=0.35)

    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    if landmarks:
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.imwrite('res.png', frame)

    
def centerface_detect(img_list):
    file = open(img_list)
    i = 0
    file = open(img_list, 'r')
    lines = file.readlines()
    for filename in lines:
        filename = filename.rstrip()
        filename = filename.replace("/am", "../..")
        frame = cv2.imread(filename)
        h, w = frame.shape[:2]
        landmarks = True
        centerface = CenterFace(landmarks=landmarks)
        if landmarks:
            dets, lms = centerface(frame, h, w, threshold=0.5)
        else:
            dets = centerface(frame, threshold=0.5)

        for det in dets:
            boxes, score = det[:4], det[4]
            # cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        # append or sth

if __name__ == '__main__':
    # filename = sys.argv[1]
    # test_image(filename)
    img_list = sys.argv[1]
    centerface_detect(img_list)
    
