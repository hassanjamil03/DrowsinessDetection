import cv2
import numpy as np
import math
import HelperFunctions as HF

class Line:
    def __init__(self, p1=0, p2=0):
        self.p1 = p1
        self.p2 = p2
"""
method 1:
1. Get image, turn to grayscale, cut image, and blur
2. Apply Canny
3. Apply filter mask to image
4. Apply Hough to find lines
5. Find average of lines on both sides to find lines that represent left and right lanes
    - Create array that contains all endpoints of hough lines for each lane
    - Make one big BoundingBox for each array 
    - Get corners of BoundingBox and create line which represents avg line of the array 
6. See if lines are past a certain point on the road or not to see if car is between lanes or not
"""

def LaneDetector(image):
    # Constants
    blur_intensity = 5
    image_subsection = 3
    min_hits = 50

    # Initializing
    left_lane = 0
    right_lane = 0

    # 1. Get image, grayscale, blur, and Canny
    cimg = image
    cimg2 = cimg
    img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (blur_intensity, blur_intensity), 0)
    img = cv2.Canny(img, 50, 200, None, 3)

    # 2. Apply filter mask and cut image
    masked = HF.region(img)
    cut = int(len(masked) * ((image_subsection-1) / image_subsection))
    img = masked[cut:]

    # 3. Perform hough transform to get lines for lane of image
    lines = cv2.HoughLines(img, 1, np.pi / 180, min_hits, None, 0, 0)
    print("len of lines = " + str(len(lines)))

    # Draw the lines
    if lines is not None:
        # For every line...
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            line = Line()
            line.p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)) + cut)
            line.p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)) + cut)
            cv2.line(cimg, line.p1, line.p2, (0, 0, 255), 3, cv2.LINE_AA)
            print(left_lane)
            # 5. Get one line for each lane, first identify whether each line is left or right lane
            right_lane, left_lane = HF.r_or_l(right_lane, left_lane, line.p1, line.p2)

    l = Line()
    r = Line()
    print(left_lane)
    l.p1, l.p2 = HF.average_line(left_lane)
    r.p1, r.p2 = HF.average_line(right_lane)

    cv2.line(cimg2, l.p1, l.p2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(cimg2, r.p1, r.p2, (0, 0, 0), 3, cv2.LINE_AA)
    HF.imshow("yeahhhhhh", cimg2)

    if len(right_lane) == 0 or len(left_lane) == 0:
        return 1
    else: return 0


"""
HF.imshow("Detected Lines (in red) - Standard Hough Line Transform", cimg)

print("left_lane = " + str(left_lane))
print("right_lane = " + str(right_lane))

x, y, w, h = cv2.boundingRect(left_lane)
cv2.rectangle(cimg, (x, y), (x+w, y+h), (0, 255, 0), 2)
HF.imshow("boundingbox image ll", cimg)

cv2.line(cimg2, (x, y), (x+w, y+h), (0, 0, 0), 3, cv2.LINE_AA)

x, y, w, h = cv2.boundingRect(right_lane)
cv2.rectangle(cimg, (x, y), (x+w, y+h), (0, 255, 0), 2)
HF.imshow("boundingbox image rl", cimg)

cv2.line(cimg2, (x, y), (x+w, y+h), (0, 0, 0), 3, cv2.LINE_AA)
HF.imshow("boundingbox+line image", cimg2)
"""
