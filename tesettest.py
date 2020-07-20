import cv2
import numpy as np
import math
import HelperFunctions as HF

# file dedicated to testing out random stuff



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
"""# Constants
blur_intensity = 5
image_subsection = 3
min_hits = 50

# Initializing
left_lane = 0
right_lane = 0

# 1. Get image, grayscale, blur, and Canny
cimg = cv2.imread("road_photos/road.jpg", 1)
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

        # 5. Get one line for each lane

        # if the line is a apart of the right lane
        right_lane, left_lane = HF.r_or_l(right_lane, left_lane, line.p1, line.p2)

print(left_lane)
print(left_lane[:, 1])
print(int(sum(left_lane[:, 1])/len(left_lane[:, 1])))

print(right_lane)
print(right_lane[:, 1])
print(int(sum(right_lane[:, 1])/len(right_lane[:, 1])))
"""
p1 = (3, 4)
p2 = (4, 6)
aryy = np.array([p1, p2]).reshape(4)
print(aryy)
test_list = np.array([[3456, 45, 6456, 745], [234, 8756, 8, 3]])
test_list = np.append(test_list, [aryy], axis=0)
print(aryy)
print(test_list)
print(sum(test_list[:, 0])/ len(test_list[:, 0]))
