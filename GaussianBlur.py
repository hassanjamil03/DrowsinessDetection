
"""
perform gaussian blur on an image
- first get image from specified folder on comp
- download image into program -> inp
- perform gaussian blur

- get inp dimensions
- create vector with same dimensions as inp, accounting for kernael -> out
- convert inp into black and white -> temp
- apply kernel and to every pixel and divide to copy result to out

- save out to specified folder on comp
"""


import cv2 as cv
import numpy as np
import math

# perform canny on image
img = cv.Canny(cv.imread("timmy.jpg", 0), 0, 50)

# perform hough transform to get lines for lane of image
lines = cv.HoughLines(img, 1, np.pi / 180, 50, None, 50, 10)
# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(img, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
cv.waitKey()
""" 
- get image
- convert to b/w
- (haar cascade / or something) to find face
- k means (k = 3) all pixels into 0, 128, or 255
- 
----------------------------------------or-----------------------------------------
perhaps use haar cascade / eigenface or something to find huge dataset of eyes
train neural network from scikit learn on this dataset
------------------------------------------or--------------------------------------
use haar cascade to find facial feutures and narrow down to eyes
do some editing similar to paper (maybe canny then threshold)
apply circle hough transofrm to look for circles in the images (pupils)
- if circle/pupil is found, means the eye is open
- if circle/pupil is not found, means eye is closed
"""
eye_detect_haar = cv.CascadeClassifier('haarcascade_eye.xml')
#eye_detect_haar.