import cv2
import numpy as np


def MiddlePoint(edge, height, width):
    start_height = height - 10
    middleWidth = width / 2
    point = []
    while start_height > 0:
        right = 0
        left = 0
        countl = 0
        countr = 0
        i = 0
        for i in range(width - 1):
            if edge[start_height][i]:
                if i < middleWidth:
                    left = left + (i)
                    countl += 1
                else:
                    right = right+i
                    countr += 1

        if countr != 0:
            right = right/countr
            cv2.circle(edge, (int(right), start_height), 2, (255, 0, 0), -1)
        if countl != 0:
            left = left/countl
            cv2.circle(edge, (int(left), start_height), 2, (255, 0, 0), -1)
        middle = 0
        if left and right:
            middle = (left + right)/2
        elif left:
            middle = (width - (middleWidth - left) + left)/2
        elif right:
            middle = middleWidth - (width - right)
        if middle:
            point.append([middle, start_height])
            cv2.circle(edge, (int(middle), start_height), 2, (255, 0, 0), -1)
        start_height = start_height - 50

    if point == []:
        return None, edge
    return point[0], edge


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


capture = cv2.VideoCapture('./video-1653143035.mp4')

while True:
    sucess, img = capture.read()
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        smoothed_img = gaussian_blur(img=gray, kernel_size=5)
        canny_img = canny(img=smoothed_img,
                          low_threshold=180, high_threshold=240)
        mid, edge = MiddlePoint(
            canny_img, canny_img.shape[0], canny_img.shape[1])
        print(mid)
        cv2.imshow("A", edge)
    except:
        cv2.imshow('img', img)
        print('error')

    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
