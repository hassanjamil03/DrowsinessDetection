import cv2
import numpy as np
import math
import dlib
from scipy.spatial.distance import euclidean as get_dist
import HelperFunctions as HF

""" 
my method 1
1.    face detect
2.    eye detection
3.    apply contours on eyes
4.    copy contours on eyes only black empty image
5.    run hough circle transform on image to see if the contour is circle or not, if its circle then its a pupil
-----------------------------------or-----------------------------
my method 2
1.    face detect
2.    eye detect, create dark image with same dimensions
3.    contours to find pupils, draw contour onto dark
4.    check if dark image has circle in it with hough circle transform
--------------------------------------or------------------------------
my method 3
- get image
- convert to b/w
- (haar cascade / or something) to find face
- crop image to face
- find eyes on face
- crop image to eyes
- k means (k = 3) all pixels into 0, 128, or 255
- use hough circle transform to find pupils
----------------------------------------or-----------------------------------------
my method 4
- haar cascade / eigenface or something to find huge dataset of eyes
- train neural network from scikit learn on this dataset
------------------------------------------or--------------------------------------
my method 5
- use haar cascade to find facial feutures and narrow down to eyes
- do some editing similar to paper (maybe canny then threshold)
- apply circle hough transofrm to look for circles in the images (pupils)
- if circle/pupil is found, means the eye is open
- if circle/pupil is not found, means eye is closed
-------------------------------------------or-------------------------------------
my method 6
- face detect
- eye detection
- apply contours on eyes
- copy contours on eyes only black empty image
- run hough circle transform on image to see if the contour is circle or not, if its circle then its a pupil
-----------------------------------------or-------------------------------------
my method 7
dlib facial landmarks
1.    get image, turn gray
2.    facial landmarks
3.    get landmark points for eyes
4.    calculate EAR
5.    see whether eye is open or closed based on EAR

"""
# method 7
def EyeDetector(image):
    img = image
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Constants
    count = 0
    min_EAR = 6

    # 1.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    print("faces found = " + str(len(faces)))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # 2.
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = predictor(gray, dlib_rect).parts()
        landmarks = [[p.x, p.y] for p in landmarks]

        # placing landmarks on image
        for y, x in landmarks:
            final_gray = HF.place_landmarks([y, x], img)

        # 3. calculating right and left eye aspect ratios (EAR)
        # for right eye: 36-41, for left: 42-47 (zero indexed)
        average_EAR = HF.EAR(landmarks)
        print("average = " + str(average_EAR))

        # 4.
        if average_EAR < min_EAR: return 1
        else: return 0




""" # method 2:
#def EyeDetect(img_name):
#    img = cv2.imread(img_name)
img = cv2.imread("pexels-photo-614810.jpeg")
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# 1.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.5, 5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    cv2.imshow("face detect", roi_gray)
    # 2.
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        roi_gray_eye = roi_gray[ey:ey + eh, ex:ex + ew]
        #cv2.GaussianBlur(roi_gray_eye, roi_gray_eye, (3,3), 0)
        cv2.imshow('roi_gray_eye', roi_gray_eye)
        cv2.waitKey()
   """     """_, roi_gray_eye = cv2.threshold(roi_gray_eye, 0, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
        cv2.imshow("threshold otsu image", roi_gray_eye)
        cv2.waitKey()"""


"""
        #dark = np.zeros((ew, eh))

        #3.
        _, pupil_contour = cv2.findContours(roi_gray_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(roi_gray_eye, pupil_contour, -1, 255, 3)
        cv2.imshow(roi_gray_eye)
        cv2.waitKey()
        cv2.destroyAllWindows()
        #4.
        #cv2.HoughCircles(dark, )
"""
