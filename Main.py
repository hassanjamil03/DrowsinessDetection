import EyeDetector as ED
import LaneDetector as LD
import cv2


def main():
    # Variables
    camera = cv2.VideoCapture(0)
    count = 0
    min_counts = 10
    hold = False

    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            break
        if (ED.EyeDetector(img) or LD.LaneDetector(img)) or (count < min_counts and hold):
            count = count + 1
        elif count >= min_counts:
            print("driver is drowsy")
            break
        else:
            count = 0


if __name__ == '__main__':
    main()
